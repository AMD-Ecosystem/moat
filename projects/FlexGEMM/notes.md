# FlexGEMM notes

## Intake screen (2026-08-27)

Upstream: https://github.com/JeffreyXiang/FlexGEMM — 148 stars, active
(`pushed_at` 2026-06-25), not archived. Author Jianfeng Xiang, the TRELLIS.2
author; MOAT already ported CuMesh from the same author.

**Recommendation: decline, reason `already-supported`.** Upstream already runs on
ROCm, by the author's design and by a merged upstream PR. Details below.

### Licence — MIT, tier 1

`utils/licenses.py check JeffreyXiang/FlexGEMM` -> `license=MIT tier=1`. Verified
by reading the file, not just GitHub's field: `LICENSE` is the verbatim MIT text,
"Copyright (c) 2025 Jianfeng Xiang (belljig@outlook.com)". `pyproject.toml` carries
`license = { file = "LICENSE" }`. Recorded as `status.json.license_spdx = "MIT"`.

Per-file checks, both clean:

- `utils/licenses.py scan-nvidia agent_space/FlexGEMM-screen` -> "no NVIDIA
  proprietary licence text". No `LICENSE_NVIDIA`-style file anywhere.
- No submodules and no vendored third-party directory. `.gitmodules` does not
  exist; the clone was taken with `--recurse-submodules` and produced nothing
  extra. The whole tree is 70-odd first-party files plus `assets/` and
  `autotune_cache.json`. The EnvGS failure mode (permissive top level over an
  unlicensed submodule) does not apply here.

Nothing about the licence is unresolved and nothing needs a clearance decision.

### The viability crux: NOT CUTLASS-bound, NOT tensor-core-shaped

The screening brief flagged a risk that FlexGEMM leans on CUTLASS or other
NVIDIA-specific GEMM infrastructure. It does not. The opposite is true, and it is
the central fact of this screen.

FlexGEMM's GEMM is **Triton**, not CUDA. The README calls it a "Triton-powered
GEMM backend" and "Triton-First Architecture"; `pyproject.toml` describes the
package as "A Cross-Platform Backend for High-Performance Sparse Convolutions" and
depends on `triton>=3.2.0` (`triton-windows` on Windows). All three algorithm
variants — Explicit, Implicit, Masked Implicit, each with a Split-K form — live in
`flex_gemm/kernels/triton/spconv/*.py`, about 3.3k lines of Triton.

A grep of the entire C++/CUDA tree for `cutlass|cublas|cusparse|cusolver|curand|
mma\.|wmma|__ldg|ptx|asm volatile|__shfl|cooperative_groups|cudnn|thrust|cub::|
__nv_|__half` returns **zero hits**. There is no NVIDIA library dependency, no
inline PTX, no warp-shuffle, no tensor-core intrinsic anywhere in the compiled
extension.

The CUDA extension is small and plain — 3,333 lines across
`flex_gemm/kernels/cuda/`, and it is not GEMM at all. It is index-and-bookkeeping
work: an open-addressing hashmap (`hash/`, 584 lines, `atomicCAS` on 32- and
64-bit keys), Z-order and Hilbert serialization (`serialize/`, 739 lines, pure
integer bit-twiddling), grid sample (`grid_sample/`, 357 lines), and sparse
neighbor-map construction (`spconv/`, ~1,400 lines). The only warp-level code in
the whole extension is one reduction in
`flex_gemm/kernels/cuda/spconv/migemm_neighmap_pp.cu`, and it already uses
`warpSize` as a runtime value rather than a hardcoded 32.

Build system is `setup.py` with `torch.utils.cpp_extension.CUDAExtension` — the
standard PyTorch path, which hipifies. There is no CMake, no bazel, no vendored
build.

So on portability grounds this project is easy. That is not the reason to decline;
the reason is that the work is already done.

### Duplicate effort — the port already exists upstream

Four independent, mutually confirming signals:

**1. A merged upstream ROCm PR.** `JeffreyXiang/FlexGEMM#18`, "Add AMD ROCm/HIP
support (2-line fix)" by ZJLi2013 (zhengjia), opened 2026-04-08 and **merged
2026-04-20** as `92ea4372`. Its two commits are `778e1dbd` ("Fix __syncwarp for
ROCm/HIP: AMD wavefront lockstep via wave_barrier") and `c8939d6a` ("Disable tf32
input_precision on AMD/ROCm (only ieee supported)"). Total diff: 8 added lines, 1
deleted, across two files. Both changes are present at current HEAD:

    flex_gemm/kernels/cuda/spconv/migemm_neighmap_pp.cu:9
      #if defined(__HIP_PLATFORM_AMD__)
        #if !defined(__syncwarp)
          #define __syncwarp(...) __builtin_amdgcn_wave_barrier()

    flex_gemm/kernels/triton/spconv/config.py:6
      allow_tf32 = not torch.version.hip

That is the entire delta the CUDA extension needed. An earlier identical PR (#17)
was closed and resubmitted as #18.

**2. `setup.py` has a first-class ROCm build path**, independent of PR #18:

    from torch.utils.cpp_extension import CUDAExtension, BuildExtension, IS_HIP_EXTENSION
    ...
    BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")   # "auto" | "cuda" | "rocm"
    if IS_HIP: archs = os.getenv("GPU_ARCHS", "native").split(";")
               cc_flag = [f"--offload-arch={arch}" for arch in archs]

`GPU_ARCHS` / `--offload-arch` is exactly the knob a ROCm build needs, and it is
already there with an explicit `BUILD_TARGET=rocm` override.

**3. The author designed for AMD from day one.** `flex_gemm/kernels/triton/
spconv/config.py` carries a dedicated `'hip'` platform config list *and* a
`'MI300X'` device config list, and both are tuned with ROCm-Triton-specific MFMA
knobs — `waves_per_eu`, `kpack`, `matrix_instr_nonkdim` — that have no CUDA
meaning. Checking the blob at every commit that ever touched that file shows the
`'hip'` block and the `MI300X` block present in the **first commit**
(`3061da44`, 2025-08-25), not retrofitted. `flex_gemm/kernels/triton/utils.py`
dispatches on `torch.version.hip` to return platform `'hip'`.

**4. Real MI300X autotuning is shipped in the repo.** `autotune_cache.json`
(2.7 MB, installed to `~/.flex_gemm/` by `setup.py`) has three top-level GPU keys:
`NVIDIA A100-SXM4-40GB`, `NVIDIA A100 80GB PCIe`, and **`AMD Instinct MI300X VF`**.
The MI300X entry holds roughly 18,500 tuned configurations across 24 kernels —
e.g. 5,452 entries for `sparse_submanifold_conv_bwd_weight_masked_implicit_gemm_
splitk_kernel`, 4,501 for the matching input kernel, 4,074 for the fwd splitk
kernel. Somebody ran the full autotuner on real MI300X hardware and committed the
results. That is beyond a compile-fix; it is performance work.

Searches that came up empty or non-authoritative:

- No `AMD-Ecosystem/FlexGEMM` and no `ROCm/FlexGEMM`. A global
  `FlexGEMM in:name` search returns no AMD-org repo. No other AMD team owns this.
- `grep -rniE 'amd|rocm|hip|gfx[0-9]' README* docs/` — the README has no "notable
  forks" section and no AMD mention; the ROCm support is in the code, not the docs.
  A docs-only grep would have missed all of this.
- Open upstream PRs: #31 (Windows/MSVC build, rwfsmith) and #29 (draft, CPU
  fallbacks, RisingRedstone). Neither is ROCm work, so there is no in-flight port
  to contribute to either.
- Upstream branches: `adaptation_for_ocnn`, `dev/all_triton`, `dev/serialize`,
  `dev/sparse_conv`, `dev/tilelang`, `main`, `uint64_hash`. None ROCm-related.
- `ATLAS-0321/FlexGEMM-rocm`, branch `rocm-port`, 1 commit ahead of upstream
  (2026-05-02): a single commit "Add ROCm/HIP kernel port for grid_sample, hash,
  serialize, spconv" that adds a **duplicated** `flex_gemm/kernels/hip/` tree —
  hipify-style `.hip` copies of every `.cu`, ~3,300 added lines, with the original
  `cuda/` tree left in place. It was never offered upstream (no PR), it is not by
  an AMD team, and its approach — forking the source tree rather than the two-line
  guard upstream took — is the one MOAT would not recommend. Not authoritative; it
  is superseded by the merged #18. Noting it only so nobody rediscovers it and
  mistakes it for prior art worth building on.
- `Cardboard-box-a/FlexGEMM-rocm` is not a GitHub fork (`parent` is null) and is
  not comparable against upstream; it appears to be a re-upload. No signal.

### Dependencies

Runtime imports across `flex_gemm/` are `torch`, `triton`, `filelock`, and stdlib
only. No MOAT project is a dependency, so `depends_on` stays empty. FlexGEMM is
itself a dependency of TRELLIS.2 (screened in parallel) — see below.

### Consequence for TRELLIS.2

TRELLIS.2 lists FlexGEMM in its dependency stack. Because upstream FlexGEMM
already supports ROCm and installs via `pip install .` with the ROCm path
auto-detected (`IS_HIP_EXTENSION`), **FlexGEMM is a satisfied dependency for
TRELLIS.2, not a blocker.** `moatlib`'s `DEP_SATISFIED_BY_DISPOSITION` treats
`already-supported` as satisfying a hard dependency, so if a person records this
decline the TRELLIS.2 selector will not wait on it. TRELLIS.2 should install
upstream FlexGEMM as-is.

The one caveat to hand to whoever ports TRELLIS.2: the MI300X autotune cache
covers submanifold-conv kernels, and the `'hip'` fallback config list is short (5
configs versus 8 for CUDA and 16-18 for the tuned A100/H100/MI300X device lists).
On a non-MI300X AMD part, or for the `sparse_conv` (non-submanifold) kernels,
Triton will autotune from that 5-config list at first run — correct, but a cold
start and possibly not peak. That is a tuning observation for the consumer, not a
port.

### Residual technical observation (not a reason to adopt)

`__builtin_amdgcn_wave_barrier()` is a scheduling barrier, not an execution or
memory barrier. In the `migemm_neighmap_pp.cu` reduction it stands in for
`__syncwarp()` between LDS read-modify-writes where lanes consume values written
by other lanes:

    if (iters_warpwise > 0 && threadIdx.x < warpSize) {
        for (int i = 0; i < iters_warpwise; i++) {
            int cur_len = warpSize >> i;
            buf[threadIdx.x] |= buf[threadIdx.x + cur_len];
            __syncwarp();
        }
    }

On CDNA this works in practice because a wavefront executes in lockstep, and the
`warpSize`-relative indexing is already wave64-correct. It is nonetheless the kind
of construct that can break under a future compiler that reorders LDS accesses
across the barrier; `__threadfence_block()` would be the stronger form. This is a
small upstream hardening opportunity, not a port, and not sufficient reason to
adopt the project. If someone wants it, it is a one-line drive-by PR, not a MOAT
pipeline run.

### Recommendation

Decline, `SKIP_REASON = already-supported`. Upstream supports ROCm today: the
enabling change is merged (#18), the build system has a ROCm target, the Triton
tuning tables include MI300X-specific MFMA configs from the first commit, and
~18,500 real MI300X autotune results ship in the repo. There is no port left to
write. Adopting it would produce a fork whose diff against upstream is, at most,
one defensive barrier change.

Recorded via `moatlib.py set-intake FlexGEMM decline --reason already-supported`.
Per the autonomy boundary this is a recommendation only — no `triage.py skip` was
run and `dispositions.json` was not touched. A person decides.

### Commands run

    python3 utils/licenses.py check JeffreyXiang/FlexGEMM
    git clone --depth 1 --recurse-submodules https://github.com/JeffreyXiang/FlexGEMM \
        agent_space/FlexGEMM-screen
    python3 utils/licenses.py scan-nvidia agent_space/FlexGEMM-screen
    gh pr list --repo JeffreyXiang/FlexGEMM --state all
    gh api repos/JeffreyXiang/FlexGEMM/branches --paginate --jq '.[].name'
    gh api repos/JeffreyXiang/FlexGEMM/forks
    gh api repos/JeffreyXiang/FlexGEMM/compare/main...ATLAS-0321:rocm-port

Read-only throughout; no fork created, no upstream write.

## Intake re-screen for the windows gate (2026-11-03, windows-gfx1151)

Why a re-screen: the 2026-08-27 decline (`already-supported`) was never ruled, and since then jeff ruled TRELLIS.2 `fork` (2026-10-29). TRELLIS.2 records `depends_on = ["CuMesh", "FlexGEMM"]` and owes all three required gates, `windows` included, so the question is no longer "does FlexGEMM run on ROCm" but "does it run on ROCm on every gate TRELLIS.2 must pass". `moatlib.py dep-doc-gaps` also lists FlexGEMM as a provider with no `## Install as a dependency` section.

**Recommendation: fork (reversing the August decline), scoped to the Windows ROCm build path plus validation on all three gates.** The Linux finding stands unchanged: upstream supports ROCm on Linux. Windows is not covered, and the dependency recipe cannot be written from a build nobody has run.

### Licence -- unchanged, MIT tier 1

`python3 utils/licenses.py check JeffreyXiang/FlexGEMM` -> `license=MIT tier=1, cleared to contribute`. Fresh clone at upstream `main` 6dd94a8 (2026-04-22): still no `.gitmodules`, no vendored third-party tree; LICENSE is the verbatim MIT text (Copyright (c) 2025 Jianfeng Xiang). `python3 utils/licenses.py scan-nvidia agent_space/FlexGEMM-screen` -> no NVIDIA proprietary licence text. Nothing unresolved.

### Upstream since August

- Default branch `main` has not moved since 6dd94a8 (2026-04-22); both PR #18 changes are still there. Repo not archived, 158 stars, last push 2026-06-25.
- `dev/all_triton` received "FlexGEMM 2.0" (PR #28, +20574/-66192, merged 2026-06-25 into `dev/all_triton`, not `main`). It still ships the same `flex_gemm/kernels/cuda/*.cu` extension and the same Windows dependency split. TRELLIS.2's `setup.sh` clones the default branch, so `main` is what a dependent installs.
- Open PRs: #31 (rwfsmith, 2026-08-21) "Fix Windows/MSVC build: template disambiguator and C++20 flags" edits the exact Windows `extra_compile_args` block in `setup.py` a Windows ROCm fix would touch -- coordinate, do not collide; #34 (explicit-GEMM blocking, no-Triton path); #29 (draft CPU fallbacks). None is ROCm or Windows-ROCm work.
- No `AMD-Ecosystem/FlexGEMM`, no ROCm-org repo.

### Why upstream does not cover the windows gate

1. **Build flags.** `setup.py`'s `platform.system() == "Windows"` branch gives `nvcc` the flags `-Xcompiler=/std:c++17 -Xcompiler=/EHsc -Xcompiler=/permissive- -Xcompiler=/Zc:__cplusplus`, unconditionally; the HIP branch only appends its `--offload-arch` list to them. On a ROCm torch, `torch.utils.cpp_extension` routes `extra_compile_args['nvcc']` straight to hipcc on Windows (`win_hip_flags` = `COMMON_HIPCC_FLAGS + COMMON_HIP_FLAGS + cflags + arch flags`, no filtering; read in the TheRock torch 2.12/ROCm 7.14 wheel here). `-Xcompiler=` is an nvcc option the clang driver does not accept; this is the fault class already recorded for gsplat/pytorch3d on Windows ("guard win32 device flags on `torch.version.hip`"). The host side additionally defaults to `cl.exe`, which cannot parse the HIP/c10-hip headers; MOAT's Windows recipe is `CC=CXX=clang-cl`. Expected delta: a `IS_HIP`-guarded Windows flag set in `setup.py`, a documented build env, and `GPU_ARCHS` instead of `native`. Not compiled here (intake does no builds), so the planner should confirm.
2. **Triton on Windows AMD.** All the GEMM work is Triton. `pyproject.toml` pulls `triton-windows` on Windows. TheRock publishes Windows torch wheels but zero Windows Triton wheels (counted on `whl-staging-multi-arch`, `v2/gfx1151`, `whl-multi-arch`: 282, 768, 2464 triton wheels, none `win`). `triton-windows` gained an AMD/TheRock HIP backend in woct0rdho/triton-windows#179 (merged 2025-12-30; README still calls AMD support "beginning"). MOAT precedent: ffpa-attn validated on windows-gfx1101 and windows-gfx1201 with `triton-windows 3.7.0.post26`, including a Triton-Windows autotuner picking a config over the 64 KB LDS limit. So viable, but unproven for FlexGEMM, and the shipped autotune cache has no entry for any RDNA part (A100 x2 and MI300X only); the `'hip'` fallback list has 5 configs tuned for CDNA MFMA knobs (`matrix_instr_nonkdim`, `kpack`), which the planner should check against RDNA3.5/wave32.
3. **No evidence on wave32 at all.** Upstream's AMD evidence is MI300X (CDNA, wave64). The deferred `flexgemm-wave-barrier-syncwarp` (jeff: defer, "revisit if FlexGEMM is adopted as a TRELLIS.2 dependency and a wave32 card shows a real fault") is exactly what wave32 validation would answer. The `migemm_neighmap_pp.cu` reduction uses `warpSize`, so it is wave-size-agnostic in indexing.

### Why fork rather than record `already-supported`

- `already-supported` would satisfy TRELLIS.2's dependency edge by disposition (`DEP_SATISFIED_BY_DISPOSITION`) and remove the record, which also silences `dep-doc-gaps`. TRELLIS.2's Windows validator would then meet the build break above with no FlexGEMM project to put the fix in, and the fix (a `setup.py` guard in an upstream repo) belongs on a FlexGEMM fork, not in TRELLIS.2.
- The `## Install as a dependency` section must be written from the provider's own validated build (DEPENDENCIES.md). Only a validator run produces that; on the fork path the validators write it for each gate.
- Upstream is active and merged #18 twelve days after it opened, so a small Windows-ROCm PR has a live destination. The Linux-side delta should be zero; the port is the Windows build plus wave32/windows evidence.

The fair case for the August answer: on Linux there is nothing to port, and if jeff is content for TRELLIS.2's windows gate to be waived or to carry a local FlexGEMM build recipe, `already-supported` is the cheaper ruling. A third option is to record `already-supported` now and let TRELLIS.2's Windows round raise a FlexGEMM intake again if the build fails; that defers the same work.

### Dependencies

Unchanged: runtime imports are torch, triton (`triton-windows` on Windows), filelock. No MOAT dependency; `depends_on` stays empty.

### Install as a dependency

Not written: no validated build exists yet (DEPENDENCIES.md forbids a guessed recipe). The first validator on each gate writes it. For Linux the expected shape is upstream's own `pip install <src> --no-build-isolation` with `GPU_ARCHS=<gfx>` against the ROCm torch.

### Commands run (re-screen)

    python3 utils/moatlib.py branch-sync --apply
    python3 utils/licenses.py check JeffreyXiang/FlexGEMM
    python3 utils/licenses.py scan-nvidia agent_space/FlexGEMM-screen
    git clone --recurse-submodules https://github.com/JeffreyXiang/FlexGEMM agent_space/FlexGEMM-screen
    git fetch --depth 50 origin dev/all_triton
    gh pr list --repo JeffreyXiang/FlexGEMM --state all
    gh pr view {20,21,28,31,34} --repo JeffreyXiang/FlexGEMM --json baseRefName,...
    gh pr diff 31 --repo JeffreyXiang/FlexGEMM
    gh pr view 179 --repo woct0rdho/triton-windows
    curl https://rocm.nightlies.amd.com/{whl-staging-multi-arch,v2/gfx1151,whl-multi-arch}/triton/
    python3 utils/moatlib.py dep-doc-gaps

Read-only; no fork created, no upstream write, no build or GPU work.

## Planning (2026-11-20, linux-gfx942 fleet job)

**Decisions for Jeff: none needed before the porter starts.** The plan ports the whole project: the CUDA extension, every Triton kernel family, the Python ops, every test script, both examples, and the FlexGEMM columns of the benchmark scripts. Nothing is deferred and no app, GUI or library layer is cut. Two points for your information, neither needing a ruling:

- The deferred `flexgemm-wave-barrier-syncwarp` (you ruled defer on 2026-10-09: revisit if FlexGEMM becomes a TRELLIS.2 dependency AND a wave32 card shows a real fault) now meets only the first half of that condition. The plan keeps your ruling and makes no change. The validators exercise the kernel on wave32 and wave64, and a fault there brings the item back to you. New evidence for that discussion: torch builds extensions with `-DHIP_ENABLE_WARP_SYNC_BUILTINS=1`, so HIP's own `__syncwarp()` (wavefront release fence, `wave_barrier`, acquire fence) is available. Upstream's `#define __syncwarp(...) __builtin_amdgcn_wave_barrier()` replaces it with the weaker form.
- The test scripts' third-party comparison columns (traveller59 spconv, torchsparse, fvdb, warpconvnet; all CUDA-only) are dropped by a validator-side driver, or spconv is satisfied by `spconv-triton`. Those columns are not FlexGEMM code, and the correctness reference is FlexGEMM's own torch-matmul `EXPLICIT_GEMM` plus a dense `torch.nn.Conv3d`. I do not count this as a scope cut. If you do, say so before the validator runs.

### Findings

- **Upstream `main` does not build against current torch on any platform.** `setup.py` forces `-std=c++17`, and torch >= 2.14 headers `#error` below C++20 (pytorch#178150, first in v2.14.0). I confirmed it here against the image's torch 2.16.0a0 (ROCm 7.16) with a preprocessor-only check, no GPU and no build: `echo '#include <torch/extension.h>' | g++ -std=c++17 -E -x c++ -I<torch>/include -I<torch>/include/torch/csrc/api/include -I<python include> -` gives `all.h:9:2: error: #error C++20 or later compatible compiler is required to use PyTorch.`, and `-std=c++20` is clean. `BuildExtension` appends the right standard itself when none is given, so the fix is to drop the explicit flags (a shared fix, no `[ROCm]` prefix). PR #18's MI300X evidence predates this (torch 2.6).
- The Windows+HIP breakage the intake predicted holds after reading the torch source. `win_hip_flags` forwards `extra_compile_args['nvcc']` unfiltered to hipcc, so the `-Xcompiler=/...` flags reach clang. `GPU_ARCHS` defaults to `native`, which fails on Windows hipcc and also suppresses `PYTORCH_ROCM_ARCH`.
- The CUDA extension needs no device-code change. Dynamic LDS is at most 32 KB (the 32-bit gray code caps V at 32), `atomicCAS` is 32/64-bit only, there are no libraries or intrinsics beyond `__ffs`/`__popc`/`warpSize`, and every `torch::empty` output is fully written.
- The Triton risk on RDNA is the fp32 `'hip'` config `B1=256,B2=256,BK=16,num_stages=2` (64 KB of operand tiles, the RDNA LDS limit). FlexGEMM's autotuner does not catch `OutOfResources`. fp16 is 32 KB.
- Upstream activity since the re-screen: PR #35 (2026-11-12, Triton 3.9 fixes on `dev/all_triton`, not `main`); issue #19 (Windows nvcc `data_ptr<T>` template disambiguator, also fixed by open PR #31; clang does not need it).

Full analysis: `plan.md`. Machine-checked surface: `surface.json` (24 components; `surface.py check` passes). `ext_type` set to `torch-extension` via `moatlib.py set-ext-type`.

### Commands run (planning)

    bash utils/orient.sh                       # selected FlexGEMM planner, linux-gfx942
    python3 utils/moatlib.py set-state FlexGEMM linux-gfx942 planning --agent planner
    git merge origin/main                      # branch-sync refused on a data/deferred.json conflict;
                                               # resolved by keeping main's file plus this branch's
                                               # moatlib-set-license-subcommand entry
    git clone https://github.com/AMD-Ecosystem/FlexGEMM projects/FlexGEMM/src   # + upstream remote, read-only
    gh pr view 18|31|35 --repo JeffreyXiang/FlexGEMM; gh pr diff 31|35 ...; gh issue list/view 14,19 ...
    git -C /var/lib/jenkins/pytorch log -S'C++20 or later compatible compiler' -- torch/csrc/api/include/torch/all.h
    g++ -std=c++17|c++20 -E (torch/extension.h)  # preprocessor only
    python3 utils/surface.py generate FlexGEMM && python3 utils/surface.py check FlexGEMM
    python3 utils/moatlib.py set-ext-type FlexGEMM torch-extension

## Porting (2026-11-20, linux-gfx942 fleet job, AMD Instinct MI300X)

**Questions for Jeff (none blocks the reviewer; the port proceeds without them):**

1. **`examples/spconv.py` is broken upstream on every platform.** It unpacks `out_feats, neignbor_cache = sparse_conv3d(...)`, but `sparse_conv3d` returns three values `(feats, coords, neighbor_cache)` since the API change, so it raises `ValueError: too many values to unpack (expected 2)` before any GPU work. This is pure Python and not ROCm-specific, so the port leaves it alone (smallest port). With the unpack corrected (agent-side copy, no fork edit) it runs on MI300X with finite output and gradients. Options: (a) leave it (current); (b) add a separate unprefixed commit fixing the example in this port; (c) mention it to upstream separately. (b) or (c) is your call, since it widens the upstream-visible change.
2. **The HIP launch-size fix is a config prune, not a kernel change.** Commit bcf7c6f drops, on HIP only, the grid_sample backward configs whose launch would reach 2^32 work-items (details below). The alternative is to change `indice_weighed_sum_bwd_input` to a 2D grid or a persistent loop, which also covers inputs where every config is too large (roughly M*V*C at or above 2^34, i.e. tens of GB of indices and gradients), but edits a shared Triton kernel that upstream tunes. I chose the prune as the smaller change. No ruling is needed unless you prefer the kernel change.

### What was done

Fork `AMD-Ecosystem/FlexGEMM`, branch `moat-port` cut from `main` 6dd94a8 (upstream head). Four commits, head **bcf7c6f5eee9f780a0ecdff4de7cecca8f73c728**, pushed:

| Commit | Title | Change |
|---|---|---|
| 06873dd | setup.py: let torch choose the C++ standard | drop `-std=c++17`, `/std:c++17`, `-Xcompiler=/std:c++17` (shared fix, plan item 1) |
| 078c4ab | [ROCm] setup.py: Windows HIP flags and default target | Windows+HIP: hipcc gets `["-O3"] + cc_flag` only; `GPU_ARCHS` default is empty on Windows so torch uses `PYTORCH_ROCM_ARCH`; `.gitignore` gains `flex_gemm/kernels/hip/` (hipify output written in-tree by every ROCm build; upstream left it untracked) |
| d45c5e2 | [ROCm] README: describe the AMD GPU build | short "AMD GPUs (ROCm)" paragraph under Installation (GPU_ARCHS, Windows triton-windows + VS prompt). No device-selection env vars. |
| bcf7c6f | [ROCm] grid_sample: skip bwd configs past the HIP launch size limit | `early_config_prune` on `indice_weighed_sum_bwd_input_kernel`, HIP only |

`/openmp` in the Windows `cxx` list was kept (plan risk 4: drop only if the Windows link proves it must go). No `.cu`/`.cuh`, spconv Triton kernel, op, test or autotune-cache change. The deferred `flexgemm-wave-barrier-syncwarp` is untouched per your ruling; every neighbour-cache test (which runs `reduce_code_kernel`) passes bit-exact on wave64.

Flag-list diff, upstream vs port, from `agent_space/flags.py` (monkeypatched `setuptools.setup`, `platform.system` patched), GPU_ARCHS unset:

```
BUILD_TARGET=cuda Linux:   cxx/nvcc lose only -std=c++17
BUILD_TARGET=cuda Windows: cxx loses /std:c++17; nvcc loses -std=c++17 and -Xcompiler=/std:c++17 (all else identical)
BUILD_TARGET=rocm Linux:   cxx/nvcc lose only -std=c++17 (--offload-arch=native kept)
BUILD_TARGET=rocm Windows: nvcc ['-O3', '-std=c++17', '-Xcompiler=/std:c++17', '-Xcompiler=/EHsc', '-Xcompiler=/permissive-', '-Xcompiler=/Zc:__cplusplus', '--offload-arch=native'] -> ['-O3']
```

### Environment (job-local, nothing in the image's shared paths)

- TheRock ROCm **10.2.0a20261022** wheels from `https://nightly.repo.amd.com/rocm/whl-next/` installed into `$FLEET_VENV` (Python 3.11): `torch==2.16.0a0+rocm10.2.0a20261022`, `amd-torch-device-gfx942`, `triton==3.9.0+gitfc8d31a5.rocm10.2.0a20261022`, `rocm`, `rocm-sdk-core`, `rocm-sdk-devel`, `rocm-sdk-libraries`, `rocm-sdk-device-gfx942` all at that version; `python -m rocm_sdk init`, `python -m rocm_sdk test` = 27 OK. 10.2.0a20261022 is the newest 10.2 build on that index (later nightlies are 10.3). The job venv's base-env symlinks to ROCm 10.1 / Triton were removed from the job's own `venv-base-links` so only 10.2 is importable.
- Build env: `PATH=$FLEET_VENV/bin:$PATH`, `ROCM_HOME=ROCM_PATH=HIP_PATH=$(python -m rocm_sdk path --root)`, `GPU_ARCHS=gfx942`, `HOME` pointed at a job-local directory (setup.py and the autotuner write `~/.flex_gemm/autotune_cache.json`; an accidental write to the shared home during the first baseline build was removed), `TRITON_CACHE_DIR` job-local, `HIP_VISIBLE_DEVICES=2` (dispatcher pin), `ulimit -c 0`.
- Extra test deps in the venv: `tqdm`, `matplotlib`.

### Build

```
# baseline, upstream main 6dd94a8:
pip install <src> --no-build-isolation --no-deps
  -> torch/csrc/api/include/torch/all.h:9:2: error: C++20 or later compatible compiler is required to use PyTorch.
# port:
utils/timeit.sh FlexGEMM compile -- $FLEET_VENV/bin/python -m pip install <src> --no-build-isolation --no-deps
  -> Successfully installed flex_gemm-1.0.0, 0 warnings; hipcc gets -std=c++20 --offload-arch=gfx942 -DHIP_ENABLE_WARP_SYNC_BUILTINS=1 -O3
```

### CUDA no-regression compile (this host has no nvcc; job-local pip toolchain)

`pip install --target <job>/cudapkgs --index-url https://download.pytorch.org/whl/cu130 torch==2.14.1+cu130 nvidia-cuda-nvcc==13.0.* nvidia-cuda-cccl==13.0.* nvidia-cuda-crt==13.0.* nvidia-nvvm==13.0.*`; `CUDA_HOME=<cudapkgs>/nvidia/cu13` (job-local `lib64 -> lib` and `libcudart.so -> libcudart.so.13` symlinks added for the link), `PYTHONPATH=<cudapkgs>`, `BUILD_TARGET=cuda TORCH_CUDA_ARCH_LIST="8.0;9.0" python setup.py build_ext` on `git archive` exports:

- upstream `main`: **fails**, same `#error C++20 or later compatible compiler is required` (torch 2.14.1).
- port bcf7c6f: **builds**, nvcc 13.0.88, 6 `.cu` with `-std=c++20`, `sm_80` + `sm_90`, `cuda.cpython-311-x86_64-linux-gnu.so` linked. Warnings only from torch headers (`#3189-D "module" is parsed as an identifier`, 126x) and 2x `#177-D variable "b" declared but never referenced` in upstream code.
- Not covered: an older CUDA torch (< 2.14, where BuildExtension appends c++17) and Windows nvcc. The behaviour follows from torch's `append_std17_if_no_std_present`; the mi355 host can confirm with real nvcc before presenting.

### GPU tests (MI300X gfx942, GPU 2, head bcf7c6f content)

Asserting scripts, run unchanged from `tests/`:

| Script | Result |
|---|---|
| `hashmap.py` | pass (RES 1..1024, 4 dtype pairs) |
| `serialize.py` | pass (z_order + hilbert, cpu + cuda, RES up to 2048) |
| `spconv/out_coords.py` | pass (HASHMAP + EXPAND_UNIQUE vs torch, RES up to 2048) |
| `spconv/neighbor_cache.py` | pass (`torch.equal` for egemm/igemm/migemm caches incl. `reduce_code_kernel`, RES up to 512) |
| `submconv/neighbor_cache.py` | pass (egemm/igemm/igemmk/migemm/migemmk, RES up to 512) |
| `grid_sample.py` | **failed at 078c4ab** in "Trilinear Bwd" RES=2048,C=8 (`Triton Error [HIP]: Code: 1, Messsage: invalid argument`); **passes at bcf7c6f**: all four tables err_max 0 per mille, err_mean 0 |

Conv correctness tables (fp16), run through the plan's B.2 driver (`agent_space/run_noext.py`, text below; it comments out the `import spconv...`/torchsparse/fvdb/warpconvnet lines and their `kernel_functions` entries, no fork edit). Pass criterion from the plan: each Triton algorithm's err_max <= 2x dense, err_mean <= 5 per mille.

- `spconv/fwd.py` (14 rows, RES 8..512): igemm/igemmk/migemm/migemmk err_max 2 per mille on every row (dense column 2 per mille; one row 1), err_mean 0 everywhere. Dense is N/A from RES=256 by the script's design. **Pass.** 483 s cold.
- `submconv/fwd.py` (7 rows, RES 8..512): all four algorithms err_max 2, err_mean 0. **Pass.** 68 s.
- `spconv/bwd.py` (14 rows): igemm/igemmk/migemm/migemmk err_mean 0 on every row; err_max 1-4 per mille up to RES=64, then 10-16 (RES=128 K=3), 9-23 (RES=256 K=3), 8-44 (RES=512 K=3; igemm 44, migemm 26, the split-K variants 8). The largest values are on the biggest K=3 weight-gradient reductions and are smallest for split-K, which fits fp16 accumulation-order effects rather than a fault; there is no CUDA table to compare. The torch dense `Conv3d` column, against the same `EXPLICIT_GEMM` reference, reads 175-4164 per mille err_max (err_mean up to 11), far worse than every Triton algorithm. The plan's criterion (err_mean <= 5 everywhere; the 2x-dense err_max rule applies to fwd) is met. **Pass.** 1359 s cold (autotuning dominates).
- `submconv/bwd.py` (7 rows, RES 8..512): egemm err_max 0; igemm/igemmk/migemm/migemmk err_max 1-13 (largest igemm 13, migemm 12 at RES=512; split-K 4), err_mean 0 everywhere. **Pass.** 382 s.
- `examples/submconv.py`: runs to completion (rc 0, 99 s cold).
- `submconv/inf_ablation.py` (rc 0, 315 s) and `submconv/train_ablation.py` (rc 0, 631 s), FlexGEMM-only, run unchanged: all rows RES 8..512 complete. Inference at RES=512,C=128: egemm 29.0 ms, igemm 4.3, igemm_splitk 4.8, migemm 3.0, migemm_splitk 2.9. Training at RES=256,C=256: egemm 36.9 ms, igemm 14.2, migemm_splitk 7.0. These are the RDNA-tuning baseline the plan asks for, not a gate.
- `submconv/inf_benchmark.py` and `train_benchmark.py` through the driver (torchsparse/fvdb/warpconvnet/spconv columns removed): flex_gemm column completes (rc 0; 5-6 s because the autotune cache was already warm). Inference RES=512,C=128 3.77 ms; training RES=256,C=256 7.35 ms.
- The ablation scripts write `conv_*.png` plots into `tests/submconv/`; moved out of the fork tree after the run (untracked test output, not a source change).

`tests/utils.py get_device_max_flops` has no MI300X entry, so the Tflops tables read N/A (cosmetic, not a port issue).

Driver `agent_space/run_noext.py`:

```
import os, re, sys
path = os.path.abspath(sys.argv[1])
third = ("spconv", "torchsparse", "fvdb", "warpconvnet")
out = []
for line in open(path).read().splitlines():
    s = line.strip()
    if re.match(r"(import|from)\s+(%s)\b" % "|".join(third), s) or re.match(r"'(%s)'\s*:\s*\(" % "|".join(third), s):
        out.append("# [removed] " + line); continue
    out.append(line)
sys.argv = [path] + sys.argv[2:]
sys.path.insert(0, os.path.dirname(path))
exec(compile("\n".join(out) + "\n", path, "exec"), {"__name__": "__main__", "__file__": path})
```

### The HIP launch-size fault (commit bcf7c6f)

`indice_weighed_sum_bwd_input_kernel` uses a 1D grid of `cdiv(C,BK) * cdiv(M,BM) * V` programs. At RES=2048 trilinear (M=4,194,304, C=8, V=8) the config `BM=1, BK=64, num_warps=2` gives 33,554,432 programs x 128 threads = exactly 2^32 work-items. Instrumented run (`agent_space/repro_gs.py`) listed every config's size; only that one reaches 2^32 and the autotuner's benchmark of it raised the error. Standalone HIP check on this GPU (`agent_space/lim/lim.hip`):

```
hipLaunchKernel       grid=33554431 block=128 total=4294967168 -> hipSuccess, last block ran
hipLaunchKernel       grid=33554432 block=128 total=4294967296 -> hipErrorInvalidConfiguration, nothing ran
hipModuleLaunchKernel grid=33554431 block=128                  -> hipSuccess
hipModuleLaunchKernel grid=33554432 block=128                  -> hipErrorInvalidValue (code 1, Triton's message)
(same at block=256: 16777215 ok, 16777216 rejected)
```

CUDA allows 2^31-1 blocks in x regardless of block size, so the config is legal there and the prune returns the full list off HIP. Warp size comes from `triton.runtime.driver.active.get_current_target().warp_size`, so wave32 parts keep twice the headroom. `examples/submconv.py` passes; `examples/spconv.py` is the upstream bug in question 1.

### Not done in this round (validator / later work, per plan)

- wave32 and Windows validation (other hosts); the Windows half of commits 078c4ab/d45c5e2 has not been compiled anywhere yet.
- `## Install as a dependency` is written by each validator from its validated build (plan E).
- RDNA autotune performance (not a gate).

## Lesson for cuda-to-rocm: HIP rejects a launch of 2^32 or more work-items per grid dimension

Belongs in `platform-limits.md`. (Rewritten in porting round 2 after review finding 4; the first version advised folding the grid into 2D without the CUDA caveat.)

Symptom: a kernel that runs on CUDA fails to launch on AMD only at large sizes; Triton reports `Triton Error [HIP]: Code: 1, Messsage: invalid argument` (often from inside autotuning, sometimes wrapped as `SystemError: <built-in method __reversed__ of list object ...> returned a result with an exception set`); `hipLaunchKernel` returns `hipErrorInvalidConfiguration`, `hipModuleLaunchKernel` `hipErrorInvalidValue`, and nothing runs.

Cause: HIP counts a launch in work-items per dimension: gridDim.x * blockDim.x must stay below 2^32, and likewise y and z. CUDA limits block counts instead: 2^31-1 blocks in x, but only 65,535 in y and z. Measurements, ROCm 10.2:

- x, MI300X (gfx942): grid 33,554,431 x 128 runs, 33,554,432 x 128 is rejected (also 16,777,215 / 16,777,216 at block 256), with both launch APIs.
- y and z, MI355X (gfx950): `grid=(1,4194303) x block=(1,1024)` runs, `(1,4194304) x (1,1024)` is rejected; `grid=(1,1,67108864) x (1,1,64)` is rejected. `hipDeviceProp.maxGridSize` reports `(2147483647, 65535, 65535)`, but y is not capped at 65,535 blocks: `grid=(1,65536) x (1,128)` and `grid=(1,33554432) x (128,1)` both run. The y and z limit is also a work-item count.
- A launch whose total over all dimensions exceeds 2^32 is fine: MI300X ran `grid=(4129777, 65) x 128` (about 2^35 work-items) correctly.

Flat 1D grids with small tiles and an extra multiplier (a neighbour count, a batch) reach the limit first.

Fix: keep the large program count on axis 0 and give it no more than (2^32 - 1) / blockDim.x programs there; put the overflow on a second axis whose block dimension is 1 (HIP allows nearly 2^32 there), and compute the program index as `pid0 + pid1 * num_programs0`, masking the programs that overrun the last row. Do this only on HIP, or keep the axis-1 count at or below 65,535: moving the large count itself to axis 1 or 2 trades the HIP failure for a CUDA one. A capped grid with a grid-stride (persistent) loop is the other option. In Triton the grid callable receives only the kernel arguments, not `num_warps`, so size axis 0 for the largest work-group (1024 work-items) unless the launch knows its config. Pruning autotune configs is not a fix on its own: past a size every config is too large and the list goes empty. In HIP C++, cap the grid and grid-stride. (FlexGEMM)

`surface.json` after this round: every component is covered (evidence per entry) except three owed items: `build:windows-rocm` (Windows compile/validation pending), `example:spconv` (question 1 above), and `doc:install-as-dependency` (validators). `pr_ready` refuses while anything is owed, so question 1 has to be answered before an upstream PR.

## Review 2026-11-20 (linux-gfx950 fleet job, AMD Instinct MI355X; reviewed head bcf7c6f5eee9f780a0ecdff4de7cecca8f73c728)

Reviewed `git diff 6dd94a8...bcf7c6f` (4 commits) with the pr-review skill. Every finding below was re-checked against the code, and the launch-size ones were reproduced on the GPU. **Verdict: changes requested.** Findings 1 and 2 are the launch-size fix. 3 to 5 are wording and record fixes.

### Findings

1. **The prune returns an empty config list for legal inputs of a few GiB, and the autotuner then crashes** (`flex_gemm/kernels/triton/grid_sample/indice_weighed_sum_bwd.py:11-20`, bcf7c6f). The configs that launch the fewest work-items are `BM=32,BK=8` / `BM=16,BK=16` (both `num_warps=2`), at about `M*C*V/2` work-items on wave64 and `M*C*V/4` on wave32. So the list is empty from `M*C*V ~ 2^33` (wave64) or `2^34` (wave32). For trilinear (V=8) that means `M*C >= 2^30`, which is 2 GiB of fp16 / 4 GiB of fp32 `grad_output` on wave64. With an empty list, `TritonPersistentCacheAutotuner.run` (`flex_gemm/utils/autotuner.py:63-66`) runs `builtins.min({})`. Reproduced on MI355X (Triton 3.9.0, warp_size 64), calling `indice_weighed_sum_bwd_input` directly (`cudacheck/repro_launch.py` in the job directory; extension stubbed, Triton kernels from the fork tree):
   ```
   bwd M=4194304 C=8    V=8 -> OK (best BM 32, BK 8)          # the tests/grid_sample.py RES=2048 case
   bwd M=1040000 C=1024 V=8 -> OK
   bwd M=1048576 C=1024 V=8 -> ValueError: min() arg is an empty sequence
   upstream 6dd94a8, same M=1048576 C=1024 V=8 -> SystemError: <built-in method __reversed__ of list object ...> returned a result with an exception set
   ```
   This is not a regression: upstream on HIP fails the same input. But the fix is incomplete. The commit body says "The remaining configs (larger BM) cover the same shapes", which is false in this range. The notes' question 2 says the gap needs "roughly M*V*C at or above 2^34, i.e. tens of GB", which is wrong for wave64 (2^33, 2 to 4 GiB) and also understates wave32 (4 to 8 GiB). Fix: change the launch so that no legal input is left without a config (see the view on question 2 below). If a prune remains as well, it must never return `[]`.
2. **The forward kernel of the same op has the same fault and no guard** (`flex_gemm/kernels/triton/grid_sample/indice_weighed_sum_fwd.py:80`). It uses the same `config.autotune_config` and the same 1D grid without the V factor: `cdiv(C,BK)*cdiv(M,BM)`. `BM=1,BK=64,num_warps=2` reaches exactly 2^32 work-items at `M*cdiv(C,64) = 2^25`. Reproduced on MI355X:
   ```
   fwd M=1048576 C=1024 V=8 -> OK (best BM 4, BK 64)
   fwd M=2097152 C=1024 V=8 -> SystemError: <built-in method __reversed__ of list object ...> returned a result with an exception set   (port and upstream alike)
   ```
   That is 2M query points with 1024 channels, a 4 GiB fp16 output. Whatever fixes finding 1 has to cover this kernel too. I checked the other Triton launch grids, in `flex_gemm/kernels/triton/spconv/*.py`: their worst config packs 16 output elements per work-item (`B1=128,B2=32,num_warps=4`), so they reach 2^32 only at `N*Co >= 2^36` (about 128 GiB of fp16 output). They are not a practical concern and are not part of this finding.
3. **The 078c4ab commit body states a false reason.** It says "torch adds its own MSVC flags to the hipcc command". In the default ninja path (`BuildExtension.win_wrap_ninja_compile`; torch 2.16 ROCm `cpp_extension.py:1246` and `1272`), a HIP build extends `common_cflags` with `COMMON_HIP_FLAGS` only, and that is what hipcc gets behind `-Xcompiler`. `COMMON_MSVC_FLAGS` reach hipcc only in the non-ninja `win_wrap_single_compile` fallback (`:1184`). The change itself is fine, because clang enables C++ exceptions by default and sets `__cplusplus` correctly, so hipcc needs none of `/EHsc`, `/permissive-` or `/Zc:__cplusplus`. Reword the body to give that reason instead.
4. **The lesson's "fold the grid into 2D" advice would break CUDA without a caveat** (`## Lesson for cuda-to-rocm: HIP rejects a launch ...`). CUDA caps gridDim.y and gridDim.z at 65,535 blocks, so moving the large tile count onto axis 1 or 2 trades the HIP failure for a CUDA one. The lesson must say that the large count stays on axis 0, or that the program loops. I also measured the per-dimension claim on MI355X (gfx950, ROCm 10.2, `cudacheck/lim/lim.hip`), and it holds: `grid=(1,4194303)x(1,1024)` runs, `(1,4194304)x(1,1024)` gives `hipErrorInvalidConfiguration`; `grid=(1,1,67108864)x(1,1,64)` is rejected. HIP's y/z limit counts work-items, not blocks: `hipDeviceProp.maxGridSize` reports `(2147483647,65535,65535)`, but `grid=(1,65536)x(1,128)` and `grid=(1,33554432)x(128,1)` both run. Add these y/z measurements and the CUDA caveat before promotion.
5. **The 06873dd Test Plan has no CUDA build, although the commit changes the CUDA flags.** This is the shared, unprefixed commit, and a CUDA maintainer will read it. Add the CUDA build from the compile check below, for example:
   ```
   BUILD_TARGET=cuda TORCH_CUDA_ARCH_LIST="8.0;9.0;12.0" python setup.py build_ext --inplace   # torch 2.14.1+cu126 and 2.11.0+cu128, nvcc 12.8
   ```

### Checked with no finding (for the porter's and Jeff's information, not findings)

- **06873dd on CUDA Linux:** built with real nvcc (below). The nvcc and g++ flag lists are identical to upstream's except for the standard, which torch now supplies: c++17 on torch 2.11, c++20 on torch 2.14.
- **06873dd on CUDA Windows**, checked by reading torch's code (no Windows nvcc host). In both torch 2.11 and 2.14, BuildExtension always gives Windows nvcc its own `-std=` (2.11 `cpp_extension.py:1014/1101` c++17; 2.14 `:1122/1211` c++20). On torch < 2.14, upstream sent nvcc that flag plus its own `-std=c++17`; the port sends torch's alone. On torch >= 2.14, upstream sent `-std=c++20` (torch), `-std=c++17` and `-Xcompiler=/std:c++17`; the port sends only `-std=c++20`, and nvcc forwards the matching `/std` to cl. For `cxx`, `append_std17_if_no_std_present` now adds `/std:c++20` (2.14) or `/std:c++17` (older), where upstream's explicit `/std:c++17` suppressed it. The port is correct there; a Windows CUDA build remains unexercised.
- **bcf7c6f arithmetic:** the prune's expression, `cdiv(C,BK)*cdiv(M,BM)*V*num_warps*warp_size < 2**32`, matches the kernel's grid lambda (`:93`) times Triton's block (`num_warps*warpSize`). The strict `<` matches the measured boundary. `warp_size` comes from Triton's AMD driver `get_current_target()`, which reads the device's `warpSize` (32 on RDNA), so wave32 keeps twice the headroom. It is unreachable on CUDA: `get_platform_name()` returns `'cuda'` when `torch.version.hip` is None, and the function returns before touching the driver. `nargs` holds `M`, `C` and `V`, because they are positional kernel arguments. The prune runs only on a cache miss with more than one config. The shipped `autotune_cache.json` has no MI300X entry for this kernel (only an empty A100 one), so no cached config bypasses it.
- **078c4ab Windows HIP flags**, checked by reading torch's code (no Windows host yet). With `GPU_ARCHS` unset on Windows, `cc_flag` is empty, so `_get_rocm_arch_flags` takes `PYTORCH_ROCM_ARCH`, or else the arch of every visible GPU. That matches both the README and the setup.py comment. With no GPU visible and no env var, torch itself would emit an empty `--offload-arch=`, which is torch's edge case and the reason the README says to set one. The `if arch` filter also fixes an empty or trailing-`;` `GPU_ARCHS` on Linux. `/openmp` stays in `cxx` (cl.exe), as the plan says. The Windows compile remains owed (`build:windows-rocm`).
- Commit hygiene: titles are 43-67 characters, `[ROCm]` is on exactly the three ROCm-specific commits, there are no trailers, `utils/jargon.py --port FlexGEMM` reports clean, and the diff adds no non-ASCII.

### Input on the porter's two questions (left open for Jeff)

1. *examples/spconv.py*: my view is (a) for this PR: the bug is pure Python, has nothing to do with ROCm, and widens a ROCm PR. Option (c), a separate one-line upstream PR or issue, is the better home if it should be fixed. Either way, `surface.json`'s `example:spconv` item needs your ruling before `pr_ready` can pass.
2. *prune vs kernel change*: given findings 1 and 2, I would prefer the kernel change. The prune leaves 2 to 4 GiB inputs failing on wave64, not "tens of GB", and it does not cover the forward kernel. One shape for the change: keep the M tiles on axis 0, move `(k, v)` to axis 1 (at most `cdiv(C,8)*V`, well under CUDA's 65,535), and keep a prune only for `BM` values whose axis-0 count reaches 2^32. Alternatively, a persistent grid-stride loop over a capped 1D grid removes the limit entirely. Both edit shared Triton kernels that upstream tunes, and both keep CUDA behaviour.

## CUDA compile check (2026-11-20, mi355 fleet host: real nvcc, no NVIDIA GPU)

Toolkit `/usr/local/cuda-12.8` (nvcc V12.8.93), host g++ 13.3.0, Python 3.11. Two job-local environments: the fleet guard allows pip only into the job's own venv, so they live at `$FLEET_VENV/cuda-new` and `$FLEET_VENV/cuda-old`. Neither touches the image's Python:

```
uv venv -p /opt/python-3.11-venv/bin/python3.11 $FLEET_VENV/cuda-new
uv pip install -p $FLEET_VENV/cuda-new/bin/python --index-strategy unsafe-best-match --index-url https://download.pytorch.org/whl/cu126 torch==2.14.1+cu126 --extra-index-url https://pypi.org/simple setuptools wheel ninja numpy
uv venv -p /opt/python-3.11-venv/bin/python3.11 $FLEET_VENV/cuda-old
uv pip install -p $FLEET_VENV/cuda-old/bin/python --index-strategy unsafe-best-match --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128 --extra-index-url https://pypi.org/simple setuptools wheel ninja numpy
```

(cu128 wheels stop at torch 2.11, so torch >= 2.14 comes from the cu126 build. torch warns about a minor version mismatch, 12.8 vs 12.6, and continues.) Each tree is a `git archive <rev>` export, built exactly as setup.py does:

```
export CUDA_HOME=/usr/local/cuda-12.8 PATH=$VENV/bin:$CUDA_HOME/bin:/usr/bin:/bin HOME=<job-local> BUILD_TARGET=cuda TORCH_CUDA_ARCH_LIST="8.0;9.0;12.0" MAX_JOBS=32
python setup.py build_ext --inplace
```

| Tree | torch | Result |
|---|---|---|
| upstream 6dd94a8 | 2.14.1+cu126 | **fails**: `torch/csrc/api/include/torch/all.h:5:2: error: #error C++20 or later compatible compiler is required to use PyTorch.` (and the ATen.h twin), from the forced `-std=c++17`. This confirms the porter's reason. |
| port bcf7c6f | 2.14.1+cu126 | **builds**: all 6 `.cu` through nvcc with `-O3 --use_fast_math -allow-unsupported-compiler -std=c++20` and `sm_80/sm_90/sm_120`, `ext.cpp` through g++ with `-std=c++20`, the `.so` links. Warnings come only from torch headers (`#3189-D "module" is parsed as an identifier`, 168x) and upstream's 3x `#177-D variable "b" declared but never referenced`. |
| upstream 6dd94a8 | 2.11.0+cu128 | builds (`-std=c++17` from setup.py) |
| port bcf7c6f | 2.11.0+cu128 | **builds**: torch appends `-std=c++17`, all other flags identical to upstream; same 3 upstream warnings |

Both port builds' `.so` load with their torch (`importlib` on the built `cuda.cpython-311-x86_64-linux-gnu.so`: 18 exported functions). There is no NVIDIA GPU here, so nothing ran on CUDA. **The port causes no CUDA compile failure.**

## Porting round 2 (2026-11-20, linux-gfx942 fleet job, AMD Instinct MI300X)

Answers the 2026-11-20 review (MI355X, head bcf7c6f). New fork head **e7d36f10fcef3efc623656ca7592361db662f04b**, one commit on top of bcf7c6f, pushed plainly to `moat-port`. No history was rewritten (no amend, rebase or force-push); the wording faults in older commit bodies are recorded under "Correction for the final history" below.

**Questions for Jeff (none blocks the reviewer):**

1. Question 1 of round 1 (`examples/spconv.py`, `surface.json` `example:spconv` owed) is still open; this round did not touch it.
2. Question 2 of round 1 (prune vs kernel change) is answered by doing the kernel change the reviewer preferred, so it needs no ruling. Round 1's estimate of where the prune left gaps ("roughly M*V*C at or above 2^34, i.e. tens of GB") was wrong: the gap started at M*C*V ~ 2^33 on wave64 (2 to 4 GiB of trilinear `grad_output`) and 2^34 on wave32, as the reviewer measured.
3. New, upstream and not ROCm-specific: the grid_sample backward kernel returns wrong gradients on the first call for each new shape key, on every platform. Registered as deferred `flexgemm-grid-sample-bwd-autotune-accumulates`; details below. Options: (a) leave it out of this port (current); (b) a separate unprefixed one-line commit in this port; (c) an upstream issue or PR. (b) or (c) widens what goes upstream, so it is your call.

### Finding 1 and 2: the launch change (commit e7d36f1)

Title `[ROCm] grid_sample: split large launches over two grid axes on HIP` (67 characters).

- New helper `flat_grid(num_programs)` in `flex_gemm/kernels/triton/utils.py`, next to `get_platform_name`. Off HIP it returns `(num_programs,)`, exactly the grid both kernels launched before. On HIP it computes `rows = max(1, cdiv(num_programs, (2**32 - 1) // 1024))` and returns `(cdiv(num_programs, rows), rows)`. Axis 0 thus never holds more than 4,194,303 programs, which keeps it under 2^32 work-items for any work-group up to 1024 (HIP's maximum, so it holds for every `num_warps` on wave64 and wave32 alike). Axis 1 has a block dimension of 1 in Triton, and HIP's axis-1 limit is also a work-item count (reviewer's MI355X measurement), so `rows` (at most a few hundred for any input below the 32-bit indexing limit) is far inside it. `rows` stays far below CUDA's 65,535 as well, although the split never runs on CUDA.
- Why the 1024 bound instead of the config's real block size: Triton 3.9 calls the grid lambda with the kernel's bound arguments only (`JITFunction.run`, `grid = grid(bound_args)`), which do not include `num_warps`, and the prune approach needed `get_current_target().warp_size` per launch. The cost is that HIP splits a launch somewhat earlier than strictly needed (at 4.19M programs instead of 33.5M for a 128-wide group), which changes only the grid shape, not the work.
- Both kernels: `block_id = tl.program_id(axis=0) + tl.program_id(axis=1) * tl.num_programs(axis=0)`. The forward kernel's store mask gains `& (block_id_m < tl.cdiv(M, BM))`, because its `offset_m` wraps modulo M, so without it the up-to-`rows - 1` programs that overrun the last row would recompute and rewrite valid rows. The backward kernel needs no new mask: an overrunning program has `offset_m >= M`, so its loads and atomics are already masked off.
- bcf7c6f's `early_config_prune` is removed, so no prune remains and no config list can go empty; the `get_platform_name` import moves to the helper.
- **CUDA is unchanged in behaviour:** `flat_grid` returns the same 1-tuple as before, so the launched grid, `program_id(0)`, the program-to-tile mapping and the autotune configs (`config.py` untouched) are identical; `program_id(1)` is 0 and the new forward mask term is always true there, because the grid covers exactly `cdiv(M,BM)*cdiv(C,BK)` programs. The compiled kernel gains one `program_id(1)` read, a multiply-add and one compare.
- Diff vs bcf7c6f: 3 files, +24 / -20. Diff vs upstream 6dd94a8 in `flex_gemm/`: 3 files, +23 / -5.

Out of scope and shared with CUDA (recorded, not changed): both kernels compute offsets in 32-bit (`offset_m * C`, `neigh_idx * C`), so tensors of 2^31 or more elements overflow on every platform. The fix covers every input below that bound; the largest shapes tested sit at it (M*C = 2^31 in the forward run).

### Reproduction before and after (MI300X gfx942, GPU 2, ROCm 10.2, torch 2.16, Triton 3.9.0)

Driver `agent_space/repro_launch.py <tree> fwd|bwd M C V [dtype]`: imports the Triton kernels of an exported tree (`git archive <rev> flex_gemm`; CUDA extension stubbed, autotune cache off), uses uint32 indices with every 7th row's first neighbour missing (0xffffffff), float32 weights and N = 65,536 rows, autotunes all 12 configs, and compares against a chunked float32 torch reference (`index_add_` for backward), with tolerance 1% of the reference maximum. For backward it checks the second call, because the first call's result is corrupted by question 3. `FORCE_CFG=<i>` replaces the config list with config i alone and prints the launched grid.

Before (launch outcome only):

```
bcf7c6f bwd M=4194304 C=8    V=8 -> launches (tests/grid_sample.py RES=2048 case)
bcf7c6f bwd M=1040000 C=1024 V=8 -> launches
bcf7c6f bwd M=1048576 C=1024 V=8 -> ValueError: min() arg is an empty sequence
bcf7c6f fwd M=1048576 C=1024 V=8 -> OK
bcf7c6f fwd M=2097152 C=1024 V=8 -> SystemError: <built-in method __reversed__ of list object ...> returned a result with an exception set
6dd94a8 bwd: all three shapes -> SystemError (same); fwd: same as bcf7c6f
```

After, e7d36f1 (autotuned):

```
bwd M=4194304 C=8    V=8 float16 -> OK (best BM16 BK16; max_abs_err 0.538 tol 0.616)
bwd M=1040000 C=1024 V=8 float16 -> OK (best BM4 BK64; 0.18 / 0.383)
bwd M=1048576 C=1024 V=8 float16 -> OK (best BM4 BK64; 0.187 / 0.383)
bwd M=2097152 C=1024 V=8 float16 -> OK (best BM4 BK64; 0.337 / 0.54)
bwd M=1048576 C=1024 V=8 float32 -> OK (best BM4 BK64; 1.14e-05 / 0.381)
fwd M=1048576 C=1024 V=8 float16 -> OK (best BM4 BK64; 0.00781 / 0.119)
fwd M=2097152 C=1024 V=8 float16 -> OK (best BM4 BK64; 0.00781 / 0.123)
fwd M=2097152 C=1024 V=8 float32 -> OK (best BM2 BK64; 1.91e-06 / 0.123)
fwd M=4194304 C=8    V=8 float16 -> OK (best BM32 BK8; 0.00391 / 0.108)
fwd/bwd M=1000 C=8 V=8 float16   -> OK (no split)
```

(The float16 backward errors are float16 atomic accumulation, about 512 contributions per row; float32 is exact to 1e-5.)

Every config forced in turn, float32, all 36 runs OK: backward M=2097152 C=1024 V=8 (grids from `(3947581, 17)` for the 64-wide-tile configs to `(4129777, 65)` for BM=1 BK=64; 67M to 268M programs, max_abs_err 1.91e-05), forward M=2097152 C=1024 V=8 (grids `(2796203, 3)` to `(3728271, 9)`, 1.91e-06), backward M=4194304 C=8 V=8 (grids `(1048576, 1)` to `(3728271, 9)`, about 5e-05). Most split grids overrun their program count (by 1, 4, 7, 13 or 49 programs; e.g. `(3728271, 9)` launches 33,554,439 programs for 33,554,432), so the overrun masking is exercised in both kernels; `(2097152, 2)` divides exactly. The `(4129777, 65) x 128` launch is about 2^35 work-items in total, which HIP accepts: the limit is per dimension. Output in `agent_space/repro-forced.txt` (job-local).

Wave32 is covered by the same 1024-work-item bound, which does not depend on wavefront size; it has not been run on a wave32 GPU in this round (the wave32 validator will).

### Project tests at e7d36f1 (MI300X)

```
source agent_space/env.sh      # HOME, TRITON_CACHE_DIR job-local; ROCM_HOME from rocm_sdk; GPU_ARCHS=gfx942; ulimit -c 0
utils/timeit.sh FlexGEMM compile -- $FLEET_VENV/bin/python -m pip install <src> --no-build-isolation --no-deps   # Successfully installed flex_gemm-1.0.0
utils/timeit.sh FlexGEMM test -- bash -c "cd <src>/tests && python grid_sample.py"     # rc 0, 73 s, autotune cache cleared first
utils/timeit.sh FlexGEMM test -- bash -c "cd <src>/examples && python submconv.py"     # rc 0, 97 s
```

- `tests/grid_sample.py`, unchanged: all four tables (Nearest Fwd, Trilinear Fwd, Nearest Bwd, Trilinear Bwd) complete through RES=2048, with flex_gemm err_max 0 per mille and err_mean 0 per mille on every row. Trilinear Bwd times: RES=2048,C=8 105.9 ms (torch 168.7 ms).
- No example uses grid_sample (`grep -rl grid_sample examples/` is empty). `examples/submconv.py` was run as a sanity check and passes; `examples/spconv.py` is question 1 of round 1. The spconv and submconv kernels and tests are untouched by this round (the diff touches only the grid_sample kernels and a new helper), so round 1's results for them stand until the validator reruns them at this head.
- `git -C projects/FlexGEMM/src status --porcelain`: clean.
- `python3 utils/jargon.py --port FlexGEMM`: clean. `python3 utils/prose.py` on the commit message: clean.

### Upstream bug: grid_sample backward is wrong on the first call for each shape (question 3)

`indice_weighed_sum_bwd_input` allocates `grad_input = torch.zeros(...)` and the kernel `tl.atomic_add`s into it, but `@triton_autotune` on the kernel has no `reset_to_zero`. On an autotune cache miss, `TritonPersistentCacheAutotuner.run` benchmarks every config on the same arguments (many runs each), all accumulating into that buffer, and then launches the best config into it once more and returns it. Measured on upstream 6dd94a8 (`agent_space/firstcall.py`, M=4096 C=8 V=8 N=1024 float32): first-call vs second-call max ratio 24,058. Adding `reset_to_zero=['grad_input'],` to the decorator makes the two calls equal (ratio 0.9999999). Every platform is affected, including CUDA; a persisted autotune cache hides it. The forward kernel stores (no accumulation), so it is unaffected. Not changed in this port (not ROCm-specific; smallest port); see question 3.

### Environment (this job)

Fresh job directory, so the round-1 environment was rebuilt the same way: TheRock `torch==2.16.0a0+rocm10.2.0a20261022`, `triton==3.9.0+gitfc8d31a5.rocm10.2.0a20261022`, `rocm[libraries,devel]==10.2.0a20261022`, `rocm-sdk-device-gfx942==10.2.0a20261022` and `amd-torch-device-gfx942==2.16.0a0+rocm10.2.0a20261022` from `https://nightly.repo.amd.com/rocm/whl-next/` into `$FLEET_VENV`; the job's own `venv-base-links` symlinks to the base env's ROCm 10.1 and Triton removed. Two traps: an unpinned `amd-torch-device-gfx942` sends pip backtracking through every 2.15/rocm10.3 nightly (pin it to torch's exact version), and without that wheel every torch GPU kernel fails with `hipErrorInvalidKernelFile` (torch's `device-gfx942` extra names it, but a plain `torch==` install does not pull it). `python -m rocm_sdk test` OK. `HIP_VISIBLE_DEVICES=2` (dispatcher pin), `ulimit -c 0`, `HOME` and `TRITON_CACHE_DIR` job-local.

## Correction for the final history

Jeff's standing rule for fork branches is no amend, no rebase and no force-push, so these commit bodies stay as they are on `moat-port`. Apply the corrected text below when the commits are prepared for the upstream pull request (a later, person-approved step). Titles are unchanged.

### 06873dd `setup.py: let torch choose the C++ standard` (review finding 5)

Replace the Test Plan with the following (rest of the body unchanged):

````
Test Plan:
Against torch 2.16 (ROCm build), the unmodified setup.py fails with the #error above. With this change:

```
pip install . --no-build-isolation --no-deps
cd tests && python hashmap.py && python serialize.py
```

CUDA (nvcc 12.8, no GPU needed): against torch 2.14.1+cu126 the unmodified setup.py fails with the same #error; with this change it builds with -std=c++20 from torch. Against torch 2.11.0+cu128 it builds with -std=c++17 from torch, with every other flag as before.

```
BUILD_TARGET=cuda TORCH_CUDA_ARCH_LIST="8.0;9.0;12.0" python setup.py build_ext --inplace
```
````

### 078c4ab `[ROCm] setup.py: Windows HIP flags and default target` (review finding 3)

Replace the first paragraph with the following (rest of the body unchanged):

```
On Windows with a ROCm build of PyTorch, the "nvcc" flag list goes to hipcc unchanged. The -Xcompiler=/... options there are nvcc syntax that clang does not accept, so give hipcc only -O3 and the target list on Windows; the host flags still reach cl.exe through the "cxx" list. hipcc needs none of /EHsc, /permissive- or /Zc:__cplusplus: clang enables C++ exceptions by default and sets __cplusplus to the language standard in use.
```

### bcf7c6f `[ROCm] grid_sample: skip bwd configs past the HIP launch size limit` and e7d36f1

bcf7c6f's body is false where it says "The remaining configs (larger BM) cover the same shapes" (review finding 1), and e7d36f1 removes everything bcf7c6f added. Recommended: squash the two into one commit with e7d36f1's title, and e7d36f1's body with its second paragraph ("The previous commit (bcf7c6f) ...") dropped and the first sentence of its third paragraph reading "This commit changes the launch shape:" instead of "This commit removes that prune and changes the launch shape instead.". If the two stay separate commits instead, replace bcf7c6f's second paragraph with:

```
Add an early_config_prune that drops, on HIP only, the configs whose launch would reach that limit. This covers the RES=2048 test case but not larger inputs: from about M * C * V = 2^33 on 64-wide wavefronts every config reaches the limit and the list is empty; the next commit replaces the prune with a launch that splits the grid. CUDA keeps every config.
```

## Review 2026-11-20 round 2 (linux-gfx942 fleet job, AMD Instinct MI300X; reviewed head e7d36f10fcef3efc623656ca7592361db662f04b)

Reviewed `git diff 6dd94a8...e7d36f1` (5 commits) with the pr-review skill, focusing on e7d36f1 and on the porter's answers to the round-1 review. Every finding was re-checked against the code and measured on the GPU. **Verdict: changes requested, one finding.** The launch change itself is correct on every input I could construct: round-1 findings 1 and 2 are resolved, and findings 3 to 5 are resolved by the "Correction for the final history" section (the corrected text is accurate; it has to be applied when the commits are prepared for the upstream PR).

### Findings

1. **`flat_grid` adds a per-launch host cost on both platforms, and the CUDA path pays it for a HIP-only fix** (`flex_gemm/kernels/triton/utils.py:28-31`, e7d36f1). Triton calls the grid lambda on every launch (`triton/runtime/jit.py:778-779`, `grid = grid(bound_args)`), so every `indice_weighed_sum_fwd`/`_bwd_input` call now runs `get_platform_name()`, which calls `torch.cuda.is_available()` (0.86 us here), and on HIP two more `triton.cdiv` calls (1.3 us each; `triton.cdiv` is a wrapped function, not plain arithmetic). `flat_grid(1000)` measures 4.0 us. End to end, a small forward call (M=1000 C=8 V=8, autotuned, 2000 calls, best of 5) goes from 28.0-28.7 us/call at bcf7c6f to 33.3 us/call at e7d36f1 on MI300X, about 16% slower. On CUDA the cost is the `is_available()` call, about 0.9 us per launch: small, but it contradicts the commit body's "on CUDA ... unchanged" and is the kind of change an upstream CUDA maintainer rejects in a ROCm PR. Upstream evaluates the platform once, at import (`get_autotune_config`, `utils.py:53`). Fix: decide the platform once at import and use integer arithmetic, for example
   ```
   def flat_grid(num_programs):
       ...
       if not _IS_HIP:
           return (num_programs,)
       rows = max(1, -(-num_programs // ((2**32 - 1) // 1024)))
       return (-(-num_programs // rows), rows)

   _IS_HIP = get_platform_name() == 'hip'
   ```
   I measured exactly that variant (`agent_space/review2/fixcheck`, not committed anywhere): 29.2-29.8 us/call, and all 12 configs still pass `bwd M=4194304 C=8 V=8`. The CUDA path then costs one module-global truth test per launch. A new commit on top (no amend), with the e7d36f1 body's correction note in "Correction for the final history" extended if the two are to be squashed.

### Checked with no finding (for the porter's and Jeff's information)

- **Round-1 findings 1 and 2 (launch size):** no prune remains, so no config list can go empty. Axis 0 holds `cdiv(n, rows) <= 4,194,303` programs because `rows >= n / 4,194,303`, so axis 0 stays below 2^32 work-items for any work-group up to 1024 (Triton's maximum: 16 warps of 64, or 32 of 32). Axis 1 has block dimension 1, and `rows <= 512` for any program count Triton's int32 `program_id` can index. The limit is a property of the HIP dispatch (32-bit work-item counts per dimension), not of the wavefront size, and FlexGEMM's configs are all `num_warps=2` (64 work-items on wave32, 128 on wave64), so the same bound holds on wave32 parts with more headroom. Not run on wave32 here (no such GPU); the wave32 validator covers it.
- **Masking of overrun programs:** backward: an overrunning program has `block_id_m >= cdiv(M,BM)`, so `offset_m >= M`, and the `grad_output`, `indices` (`other=0xffffffff`, so `valid_mask` false) and `weight` loads and the atomic are all masked (`indice_weighed_sum_bwd.py:43-56`). Forward: `offset_m` wraps modulo M (`indice_weighed_sum_fwd.py:41`), so loads stay in bounds, and the new `block_id_m < tl.cdiv(M, BM)` term (`:63`) stops the store. Without it the overrun would rewrite wrapped rows with identical values (benign, but racy); the mask is right.
- **CUDA kernel behaviour:** off HIP `flat_grid` returns `(num_programs,)`, the same 1-tuple as upstream; `program_id(1)` is 0 and the new store term is always true because the grid is exactly `cdiv(M,BM)*cdiv(C,BK)`. `config.py` and the autotune keys are untouched. Apart from finding 1, CUDA behaviour is unchanged.
- **32-bit indexing claim in the notes:** with every config having `BM*BK >= 64` and V <= 64 (V is 1 or 8 from the op), any input whose tensors stay below 2^31 elements gives fewer than 2^31 programs, so `block_id` cannot overflow before the pre-existing (shared with CUDA) offset overflow. The notes' bound is right.
- **Lesson `HIP rejects a launch of 2^32 or more work-items ...`:** the CUDA caveat (keep the large count on axis 0, or keep axis 1 at or below 65,535), the y/z measurements and the "pruning is not a fix" point are now correct. "The grid callable receives only the kernel arguments, not num_warps" checks out against Triton 3.9 `jit.py:779`. The MI300X total-size claim is confirmed and extended: `grid=(4129777, 65) x 128` (2^35 work-items) and `grid=(4177984, 257) x 128` (about 2^37) both ran correctly here. Ready to promote.
- **Deferred `flexgemm-grid-sample-bwd-autotune-accumulates`:** reproduced (`agent_space/firstcall.py`, M=4096 C=8 V=8 N=1024 float32): first/second call max ratio 26,971 on 6dd94a8 and 26,339 on e7d36f1. Upstream, not ROCm-specific, unchanged by the port; the ruling stays with Jeff.
- **Porter's driver** (`agent_space/repro_launch.py`) tests what it says, but its tolerance (1% of the reference maximum) is loose enough to hide a missing tile in backward, and in an autotuned forward run the benchmark runs of other configs fill the same output buffer, which would mask a config that skips tiles. So I wrote a stricter driver (below) and checked it with a negative control.
- Commit hygiene: e7d36f1's title is 67 characters with `[ROCm]`, no trailer, AI-agent disclosure present, `python3 utils/jargon.py --port FlexGEMM` clean, no non-ASCII added lines. `git -C projects/FlexGEMM/src status --porcelain` clean. The installed `flex_gemm` in the venv matches the e7d36f1 export (`diff -rq`, ignoring build output).

### GPU runs (MI300X gfx942, GPU 2 via the dispatcher's HIP_VISIBLE_DEVICES=2, TheRock ROCm 10.2.0a20261022, torch 2.16.0a0, Triton 3.9.0)

```
source agent_space/env.sh
git -C projects/FlexGEMM/src archive e7d36f1 flex_gemm | tar -x -C agent_space/review2/tree
```

Previously failing shapes, through the wrappers with autotuning (`agent_space/repro_launch.py <tree> <fwd|bwd> M C V float16`, absolute tree paths):

```
                         e7d36f1                                   bcf7c6f                    6dd94a8 (upstream)
bwd M=1048576 C=1024 V=8 OK (BM4 BK64; err 0.187, tol 0.383)       ValueError: min() arg is   SystemError (__reversed__ ...)
                                                                   an empty sequence
fwd M=2097152 C=1024 V=8 OK (BM4 BK64; err 0.00781, tol 0.123)     SystemError (__reversed__) SystemError
bwd M=4194304 C=8    V=8 OK (BM16 BK8; err 0.478, tol 0.616)       OK (BM16 BK16)             SystemError
float32 at e7d36f1: bwd 1048576/1024/8 err 1.14e-05; fwd 2097152/1024/8 err 1.91e-06; bwd 4194304/8/8 err 6.1e-05 (all OK)
```

Every config forced, strict driver `agent_space/review2/allcfg.py <tree> <fwd|bwd> M C V [cfgs|all]`: launches the raw JIT kernel once per config with the `flat_grid` grid (no autotuner, so no benchmark run touches the output), float32, forward output prefilled with NaN, backward output fresh zeros, float64 torch reference, tolerance 1e-5 x reference max, and it counts never-written outputs. Negative control: a copy of the e7d36f1 tree with `block_id = tl.program_id(axis=0)` restored (`agent_space/review2/sabotage`) fails `fwd M=4194305 C=64 V=8 cfg2` with 134,217,728 NaN outputs; the real tree passes it.

```
SUMMARY bwd M=1048576   C=1024 V=8: 12/12 configs OK   grids up to (4067204, 33), overrun up to 7
SUMMARY fwd M=2097152   C=1024 V=8: 12/12 configs OK   grids up to (3728271, 9)
SUMMARY bwd M=4194304   C=8    V=8: 12/12 configs OK   (1048576, 1) to (3728271, 9)
SUMMARY bwd M=2097152   C=1024 V=8: 12/12 configs OK   up to (4129777, 65), overrun 49
SUMMARY fwd M=270003    C=1000 V=8: 12/12 configs OK   partial last M tile, odd num_k (BK=8)
SUMMARY bwd M=270003    C=1000 V=8: 12/12 configs OK
SUMMARY fwd M=4194303   C=64   V=8: cfg2 OK            (4194303, 1): last unsplit size
SUMMARY fwd M=4194304   C=64   V=8: cfg2 OK            (2097152, 2): first split size
SUMMARY bwd M=134217728 C=8    V=8: cfg2, cfg11 OK     (4177984, 257) x 128, 2^30 programs, overrun 64
SUMMARY fwd M=134217728 C=8    V=8: cfg2, cfg11 OK     (4067204, 33)
```

Max abs errors: 1.9e-06 forward everywhere; backward 2.1e-05 to 6.9e-05 (C=1024 / C=8), 1.7e-03 at M=134217728 (about 16,000 float32 atomic contributions per row, tolerance 3.9e-03). Full log: `agent_space/review2/allcfg.log` (job-local).

`tests/grid_sample.py`, unchanged, against the installed e7d36f1 build with the autotune cache off (`FLEX_GEMM_USE_AUTOTUNE_CACHE=0 FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE=0 python grid_sample.py` in `src/tests`): rc 0, all four tables (Nearest/Trilinear Fwd/Bwd) through RES=2048, all 72 err_max/err_mean cells 0 per mille.

Launch overhead (`agent_space/review2/launch_overhead.py <tree>`, finding 1): bcf7c6f 27.8-28.7 us/call, e7d36f1 33.3, e7d36f1 with the import-time platform check 29.2-29.8.
