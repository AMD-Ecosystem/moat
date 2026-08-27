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
