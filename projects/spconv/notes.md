# spconv notes

## Disposition: SKIP (cant-port) -- 2026-06-11, planner

spconv's GPU kernels are not static `.cu` files; they are emitted at build time
by `cumm` (FindDefinition/cumm), a CUTLASS-derived CUDA GEMM/implicit-conv code
generator driven by `pccm` Python meta-programming. Build is `PCCMExtension`
(setup.py), NOT a torch `CUDAExtension`, so the scaffolded `torch-extension`
ext_type is wrong and hipify has nothing to translate. Everything under
`spconv/csrc/**` is `.py` meta-programs; only 1 real `.cu` in the repo
(`example/libspconv/main.cu`, an example).

Porting == porting cumm's entire CUTLASS/CuTe-style tensor-core codegen backend
to ROCm (Volta/Turing/Ampere `mma.sync` tensorop tiles + inline-PTX paths via
`gemm_algo_can_use_ptx`). PORTING_GUIDE: CUTLASS does not port to ROCm; must be
reimplemented against Composable Kernel. That is unbounded and out of MOAT
single-repo scope. Not a `depends_on cumm` because cumm is itself the
unbounded CUTLASS-codegen port, not a scaffoldable portable dependency.

No authoritative AMD/ROCm spconv port exists (no upstream AMD branch/PR, no
ROCm-org standalone, no published `spconv_rocm` despite the hypothetical name
in PORTING_GUIDE line 120). cumm has no ROCm backend either.

Recorded via: `python3 utils/triage.py skip traveller59/spconv --reason cant-port`.
See plan.md for the full evidence and the (separate, not-planned) `spconv_rocm`
AMD-native alternative if revisited.
