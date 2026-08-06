# NATTEN -- porting plan (lead platform: linux-gfx90a)

## Project
- Name: NATTEN (Neighborhood Attention Extension)
- Upstream: https://github.com/SHI-Labs/NATTEN
- Default branch: main
- Analyzed at HEAD: 92750c3 ("cutlass-fna: 64b strides (#337)"), release line 0.21.x
- License: MIT (kernels under csrc/ are largely NVIDIA BSD-3-Clause, copied/adapted from CUTLASS + xFormers)

## Disposition: SKIP (cant-port)

NATTEN's compiled backend (`libnatten`) is, by the maintainer's own design as of the 0.21 line,
**100% CUTLASS/CuTe**. There is no naive / GEMM-free / CPU CUDA path left to port. Per PORTING_GUIDE
(2026-05-30 changelog: "CUTLASS does NOT port to ROCm and never will"), a CUTLASS->ROCm port or shim
is out of scope, and the only correctness-first first slice that MOAT normally relies on (a naive
pointwise kernel set) has been deleted upstream. Separately, NATTEN **already runs on AMD GPUs today**
via its Flex Attention backend (pure PyTorch `torch.nn.attention.flex_attention`, which lowers to
Triton on ROCm) -- that path is explicitly ROCm-aware in the source. So the high-value, tractable work
that MOAT exists to do is absent here, and the device-agnostic path that does exist needs no porting.

Recorded via: `python3 utils/triage.py skip SHI-Labs/NATTEN --reason cant-port --note "libnatten is 100% CUTLASS/CuTe (Ampere FNA/FMHA + Hopper wgmma + Blackwell); naive/CPU CUDA kernels deleted upstream (CHANGELOG 'Dropped unfused / CPU backends in libnatten'); CUTLASS does not port to ROCm per guide; AMD already works via the Flex/Triton backend."`

## Existing AMD support: ROCm-supported via Flex backend (no C++ port needed)
- `src/natten/utils/device.py:33` defines `is_rocm()` (checks `torch.version.hip`).
- `src/natten/backends/configs/checks.py:682-689`: the Flex backend's `can_run_flex_attention` explicitly
  allows tensors on a ROCm device (`is_rocm(query.device)`) plus CPU; only Flex does this.
- `choose_backend` / `choose_fmha_backend` (`src/natten/backends/__init__.py:88-148`) try
  Blackwell-FNA -> Hopper-FNA -> CUTLASS-FNA -> Flex. On a ROCm torch the first three are
  unreachable (see "Why the C++ extension cannot run on AMD" below), so NATTEN auto-selects the
  Flex path on AMD. Flex is ordinary device-agnostic PyTorch; it is not a CUDA port and gains nothing
  from a HIP translation.
- Conclusion category: "AMD already supported (via a Triton/PyTorch path), and the CUDA/C++ surface is
  CUTLASS-only -> not portable." This is the explicit PORTING_GUIDE skip pattern, not "OpenCL-only, port anyway".

## Build classification: torch-extension (Strategy B would apply IF it were portable)
Evidence:
- `setup.py:36-39` imports `torch` and `torch.utils.cpp_extension` (`LIB_EXT`); `setup.py:521` registers
  `Extension("natten.libnatten", [])` with a custom `BuildExtension(build_ext)`.
- The extension is built by shelling out to **CMake** (`setup.py:380-512`), not `CUDAExtension`. The
  CMake project is `csrc/CMakeLists.txt:4` `project(natten LANGUAGES CXX CUDA)`. It deliberately does
  NOT use `find_package(Torch)` (csrc/CMakeLists.txt:49-54, "I'm going to stop finding torch through
  cmake because of caffe") and instead links libtorch by include/lib paths
  (csrc/CMakeLists.txt:207-208: `c10 torch torch_cpu torch_python cudart c10_cuda torch_cuda`).
- So it is a Torch extension (Strategy B family) whose device build is plain `nvcc` via CMake, NOT
  Torch's AOT hipify. A real port would therefore be closer to "Strategy A applied to a Torch-linked
  CMake project" than to the automatic `CUDAExtension` hipify. Moot here, because it is unportable.

## Why the C++ extension cannot run on AMD (even if compiled)
- `setup.py:80-117` only builds `libnatten` when CUDA is detected; `setup.py:92-97` reads
  `torch.version.cuda` and asserts `CUDA_VERSION >= [12,0]`. On a ROCm torch `torch.version.cuda is None`,
  so the build aborts. `setup.py:510-512`: "Libnatten is CUDA only now."
- Runtime gating is by CUDA compute capability: `src/natten/utils/device.py:41-50` `get_device_cc()`
  returns 0 unless `torch.version.cuda` is truthy; `is_cuda()` is False on ROCm. Every CUTLASS backend's
  `can_run_*` first checks `is_cuda(...)` and a specific `get_device_cc()` (SM50+ for Ampere FNA, ==90
  for Hopper, in {100,103} for Blackwell). All are unsatisfiable on AMD by construction.

## CUDA surface inventory (for the record; this is exactly what makes it unportable)
- 155 C/C++ device source+header files under `csrc/`; **51 of them directly `#include <cutlass/...>`
  or `<cute/...>`, and 138 of the 155 reference `cutlass`/`cute` (transitively, via project headers)**.
  Verified at HEAD 92750c3: `grep -rl '#include <cutlass|#include <cute' csrc | wc -l` = 51;
  `grep -rl 'cutlass\|cute' csrc | wc -l` = 138. Whole-tree `csrc/` is 71,777 LOC. The 17 files that
  never name cutlass/cute are pure C++ glue (helpers.h, natten.cpp dispatch, type aliases), not kernels.
- Kernel families (all CUTLASS/CuTe), with build-target counts per setup.py "default" autogen policy:
  - `reference` (2 splits): CuTe reference attention -- `csrc/include/natten/cuda/reference/fna_reference_forward.hpp`
    is `#include <cute/tensor.hpp>` + `<cutlass/cutlass.h>`; the .cu is `#ifdef NATTEN_WITH_CUTLASS` and
    otherwise `TORCH_CHECK(false, "libnatten not compiled with CUTLASS.")` (reference_forward.cu:36-38,191-192).
    This is NOT the old naive reference; it is CuTe.
  - `fna` (64 splits) + `fmha` (6): CUTLASS-2.x memory-efficient-attention style (xFormers lineage):
    `csrc/include/natten/cuda/fna/{kernel_forward.h,kernel_backward.h,gemm/mma_from_smem.h,gemm_kernel_utils.h}`,
    `na_utils.cuh`; entry via `csrc/include/natten/cuda/utils/cutlass.cuh` (a `CUTLASS_GLOBAL` device-kernel wrapper).
  - `hopper-fna`/`hopper-fmha` (+bwd): SM90 native -- warpgroup MMA (wgmma), TMA, mbarrier, CuTe.
  - `blackwell-fna`/`blackwell-fmha` (+bwd): SM100/103 native.
  - FNA+Hopper-FNA headers alone contain 581 lines matching wgmma/cp.async/mbarrier/tma/cute::/cutlass::arch::Sm*/tensor-op/warpgroup.
- Library / infra: links `cudart`, `c10_cuda`, `torch_cuda`; `-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1`,
  `--use_fast_math`, `--extended-lambda` (csrc/CMakeLists.txt:84-85,148). Hard submodule dep on
  `third_party/cutlass` (NVIDIA/cutlass, .gitmodules). No cuBLAS/cuFFT/cuRAND/cuSPARSE/Thrust/CUB
  (it is all hand-written CuTe), so there is no library-swap shortcut.
- Mapping each to ROCm: not a HIP-spelling exercise. CuTe/CUTLASS has no HIP equivalent; per the guide
  these kernels would have to be **reimplemented from scratch against Composable Kernel / ck_tile**,
  understanding the FNA tiling/epilogue/masking first. That is a multi-month kernel-engineering project
  (4 distinct kernel families x fwd+bwd x SM-generation-specific code), not a MOAT-style mechanical port,
  and it would duplicate effort that AMD's own attention stacks (CK fMHA, AITER, FlexAttention/Triton)
  already cover.

## Risk list (why even a "just the reference kernel" slice is not a tractable first slice)
- The reference kernel is itself CuTe (`fna_reference_kernel` is templated on `cute::` tensors/layouts);
  it is not a plain index-math CUDA kernel. Porting it means porting a slice of CuTe -> infeasible per guide.
- wave64 (gfx90a CDNA2) surface is irrelevant because nothing compiles to begin with; the blocker is
  upstream of any warpSize/`__shfl`/`__ballot` concern.
- No texture/surface, no rule-of-five resource-handle, no OOB-neighbor, no 256B-pitch fault classes apply
  (no textures, no raw neighbor-gather kernels survive) -- the sole blocker is CUTLASS/CuTe itself.

## File-by-file change list
None. READ-ONLY analysis. No fork created, no source edited. (A port is not warranted.)

## Build commands (gfx90a)
Not applicable -- the project is being skipped. For the record, the upstream build is:
`NATTEN_CUDA_ARCH=<sm> pip install -e . -v` (CMake-driven via setup.py), which aborts on a ROCm torch
because `torch.version.cuda is None`.

## Test plan
NATTEN ships a real pytest suite under `tests/` (test_fna.py, test_fmha.py, test_fmha_varlen.py,
test_hopper_fna.py, test_blackwell_fna.py, test_flex.py, test_token_permute.py, test_attn_merge.py,
test_compute_delta.py, test_torch_compile.py, test_varlen_recompile.py, test_log.py), comparing each
backend against a reference. Relevant facts for MOAT:
- All libnatten/CUTLASS tests are guarded by `skip_if_*` decorators in `src/natten/utils/testing.py`
  that require `_IS_CUDA_AVAILABLE` and `HAS_LIBNATTEN` (and specific `get_device_cc()` values). On a
  ROCm host these all **skip**, so there is no GPU correctness signal a port could earn here.
- The Flex tests (`test_flex.py`, gated `skip_if_flex_is_not_supported` -> `_FLEX_SUPPORTED and cc>=70`)
  exercise the only AMD-runnable path, but `get_device_cc()` returns 0 on ROCm so even those skip; and
  in any case Flex is upstream PyTorch, not NATTEN CUDA code, so passing it would not validate a port.
- Net: there is no GPU test that a HIP port of libnatten could make pass without first reimplementing
  CUTLASS in CK -- which the guide forbids as a "port".

## Inter-project deps (set-deps)
None set. NATTEN does not depend on any other MOAT project, and -- because it is being skipped -- no
MOAT project should be made to `depends_on` NATTEN. (Its only external code dep is the NVIDIA/cutlass
submodule, which is the reason for the skip.)

## Open questions / revisit conditions
- Revisit ONLY if upstream reintroduces a non-CUTLASS GPU kernel path (e.g. a Triton or plain-CUDA
  reference), or if MOAT's scope expands to ground-up CK/ck_tile attention-kernel authoring (a
  dedicated multi-kernel rewrite project, not a translation). Until then NATTEN is a `cant-port`.
- If a future requester specifically wants NATTEN's neighborhood-attention masking on AMD, the
  productive route is the existing Flex/Triton backend, or an AMD-native CK fMHA + neighborhood mask,
  tracked as its own kernel-authoring effort -- not as a CUDA->HIP port of this repo.
