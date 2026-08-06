# spconv -- porting plan / non-portability determination

## Project
- Name: spconv (SpConv: Spatially Sparse Convolution Library)
- Upstream: https://github.com/traveller59/spconv
- Default branch: master
- Author: Yan Yan (traveller59)

## Decision: SKIP -- cant-port (out of MOAT scope)

spconv has no static CUDA kernel sources to port. Its entire GPU compute
surface (sparse GEMM and implicit-GEMM sparse convolution, forward/backward
and backward-weights) is GENERATED AT BUILD TIME by `cumm`
(https://github.com/FindDefinition/cumm), a CUTLASS-derived CUDA GEMM/conv
code generator driven by `pccm` (a Python C++ meta-programming framework).
Porting spconv to ROCm is equivalent to porting cumm's whole CUTLASS/CuTe-style
tensor-core codegen backend to ROCm/Composable Kernel -- an unbounded effort
that PORTING_GUIDE explicitly scopes out ("CUTLASS does NOT port to ROCm and
never will... reimplement against Composable Kernel"). No authoritative AMD or
ROCm spconv port exists upstream to validate-and-improve instead.

Disposition recorded: `utils/triage.py skip traveller59/spconv --reason cant-port`.

## Existing AMD support assessment (done first, per PORTING_GUIDE)
- Upstream-docs grep `grep -rniE 'amd|rocm|hip|gfx[0-9]' README* docs/`: NO hits.
  spconv ships only `-cuXXX` PyPI wheels (cu102..cu126); no AMD/ROCm wheel,
  branch, or "notable forks / AMD support" section.
- Web search ("spconv ROCm", "spconv AMD GPU", "spconv HIP", "spconv MI300",
  "spconv_rocm"): no authoritative ROCm port. PORTING_GUIDE line 120 lists a
  hypothetical `spconv` -> `spconv_rocm` dependency substitute, but no such
  package/repo is actually published; AMD's rocm3d catalog does not provide one.
  cumm (the kernel engine) has NO AMD/ROCm/HIP support either.
- No ROCm-org standalone port, no upstream rocm/hip branch, no open AMD PR.
- Conclusion: no existing AMD effort -- authoritative or community -- to adopt.

## Build classification: NEITHER Strategy A nor Strategy B
The scaffolded `ext_type: torch-extension` is INCORRECT. Evidence:
- `setup.py` builds via `PCCMExtension(cus, "spconv/core_cc", ...)` from
  `pccm.extension` (setup.py:15-16, 211-218). There is NO `CUDAExtension`,
  no `torch.utils.cpp_extension`, no `find_package(Torch)`. Torch is a runtime
  consumer of the built `core_cc` module, not the build driver.
- The "C++ sources" under `spconv/csrc/**` are all `.py` pccm meta-programs
  (e.g. `indices.py` declares kernels via `@pccm.cuda.cuda_global_function`
  and `code.raw(...)` C++/CUDA strings). Only ONE real `.cu` exists in the
  whole repo (`example/libspconv/main.cu`, a usage example), and ZERO static
  `.cc`/`.cpp` kernel files.
- The build classes spconv as a THIRD kind: a pccm Python-codegen native
  extension. Strategy B (torch hipify) has nothing to translate -- hipify
  operates on existing `.cu`/`.cuh`; here the `.cu` text only exists after
  cumm emits it, and it is emitted as NVIDIA-tensor-core CUDA.

## Kernel / codegen surface and fault classes
- GEMM/conv params (spconv/core.py): `SHUFFLE_SIMT_PARAMS`,
  `SHUFFLE_VOLTA_PARAMS` `TensorOp((8,8,4))`, `SHUFFLE_TURING_PARAMS`
  `TensorOp((16,8,8))`, `SHUFFLE_AMPERE_PARAMS`, and the matching
  `IMPLGEMM_*` conv lists. These are literal NVIDIA `mma.sync` fragment
  shapes -- the CUTLASS instruction tiles. The generators
  (`GemmAlgo.Volta/Turing/Ampere`) live in cumm and emit CUDA tensor-core
  (and inline-PTX) code.
- Arch dispatch is NVIDIA-CC-keyed: `convops.py` selects tensorop paths via
  `desp.tensorop[0] > 0` and `CompileInfo::gemm_algo_can_use_ptx(min_arch, arch)`.
  This is the CC-collision fault class (PORTING_GUIDE: gfx90a reports major==9
  == Hopper) AND an inline-PTX path hipcc cannot assemble.
- A `Simt` (non-tensor-core) GEMM path exists, but it is still cumm-generated
  CUDA, NOT a hand-written portable kernel; it does not provide a hipify-able
  fallback independent of cumm.
- `CUMM_CPU_ONLY_BUILD` produces a CPU-only library (no GPU GEMM at all), so it
  is not a GPU SIMT escape hatch.

## Why this is out of scope (the honest determination)
A correctness-first mechanical HIP port is impossible here because there is no
mechanical CUDA-to-HIP surface: the CUDA only exists as cumm's emitted output.
The real deliverable would be an AMD/HIP backend for cumm's pccm GEMM/conv
generator (LDS/MFMA tiling, CK or rocWMMA fragment lowering, an AMD epilogue
and iterator family, AMD arch dispatch), i.e. reimplementing a CUTLASS-class
library through a new codegen path. PORTING_GUIDE (Changelog 2026-05-30)
forbids a CUTLASS->ROCm port/shim and prescribes a CK reimplementation per
kernel -- which for an entire auto-generated GEMM+implicit-conv engine is
unbounded and far outside a single-repo MOAT port. This is not a `depends_on
cumm` either: cumm is itself the unbounded CUTLASS-codegen port, not a
scaffoldable portable dependency.

## If revisited later (not planned now)
The tractable, in-scope alternative is NOT porting spconv/cumm source but
providing a `spconv_rocm`-style drop-in that backs spconv's public sparse-conv
op dispatch with AMD-native kernels (Composable Kernel grouped/implicit-GEMM
conv, or MIOpen), validated through spconv's own test suite. That is a new
AMD-native library effort, separate from MOAT's "port the CUDA in this repo"
model, and should be its own scoped project if pursued.

## Test plan (for reference, were it portable)
Upstream tests live in `test/` (`test_conv.py`, `test_all_algo.py`,
`test_multi_impl.py`) plus `example/mnist_sparse.py`; `test_before_push.sh`
runs `pytest ./test` then the fp16 mnist example. These compare CPU vs Native
vs ImplicitGemm forward/backward. They would be the GPU validation gate IF a
ROCm backend existed; they cannot run without one.

## Open questions
- None blocking the decision. The determination is firm: no portable surface
  in this repo without an AMD codegen backend for cumm.
