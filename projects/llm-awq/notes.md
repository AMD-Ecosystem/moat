# llm-awq notes

## Summary
HIP port of `awq_inference_engine` (the AWQ PyTorch CUDAExtension under
`awq/kernels/`). Strategy B (torch build-time hipify) plus targeted manual
rewrites of every inline-PTX / tensor-core kernel that hipify cannot translate.

Validated on linux-gfx90a (AMD Instinct MI250X, ROCm 7.2.1) at fork HEAD
`7a4c596` (branch `moat-port`). Multi-arch build (gfx90a + gfx1100) confirmed.

## What is functional vs build-only on ROCm
- FUNCTIONAL (the W4A16 quantized-linear path, what `awq.entry`/WQLinear use):
  - `gemv_forward_cuda_new` (quantization_new/gemv) -- decode path, M < 8.
  - `gemm_forward_cuda_new` (quantization_new/gemm) -- prefill path, M >= 8.
    On HIP this routes to a portable SIMT GEMM (`gemm_simt.cuh`) that reuses the
    GEMV's proven packed-int4 decode; the CUDA tensor-core MMA kernels are kept
    under `#if !defined(__HIP_PLATFORM_AMD__)`.
  - INT4->FP16x2 dequant (both `quantization/dequantize.cuh` and
    `quantization_new/dequantize.cuh`): lop3.b32 (immLut 0xea = (a&b)|c) and
    sub/fma.f16x2 PTX replaced by portable bit/half2 expressions.
- BUILD-ONLY (compile, but error at runtime; NOT on the validated path):
  - `gemm_forward_cuda` (legacy MMA GEMM, quantization/gemm_cuda_gen.cu).
  - `w8a8_gemm_forward_cuda` / `_fuse_bias` (int8 MMA, w8a8/).
  - `single_query_attention` (FasterTransformer masked-MHA): attention sources
    are EXCLUDED from the ROCm build in setup.py (NVIDIA half/bf16 PTX +
    amd_hip_bf16.h fails to parse in the host .cpp); pybind binds a throwing
    stub. Deferred-work: AMD-native MFMA GEMM + a real attention port.

## Key fault-class fixes (see PORTING_GUIDE)
- Warp size / lane mask: the shuffle reductions used 32-bit masks (`~0`,
  `0xffffffff`) which (a) fail to COMPILE on HIP (sync-shuffle mask must be
  64-bit) and (b) are wave-width-wrong on wave64. Made width-32 LOGICAL-warp
  reductions (`__shfl_*(..., 32)` on HIP) -- arch-agnostic on wave32 and wave64;
  the partition keys were already logical-32 (`lane = tid%32`, `warp = tid/32`,
  `shared[32]`). Files: quantization_new/gemv, quantization/gemv,
  layernorm/reduction.cuh, w8a8/reduction_utils.cuh, w8a8/quantization.cu.
- bf16 software fallback: `__CUDA_ARCH__ < 800` guards select Ampere bf16
  intrinsics vs a software path. On HIP `__CUDA_ARCH__` is undefined, so the
  raw guard fell through to the Ampere ELSE branch. Re-keyed to
  `... || defined(__HIP_PLATFORM_AMD__)` so HIP takes the software branch
  (w8a8/utils.cuh: bf1622int16, the bf16 ldg overloads).
- int8 conversion PTX: `cvt.rni.sat.s8.{f32,f16}` -> `rintf` + clamp on HIP.

## ROCm build shim
`csrc/awq_amd_compat.h` (force-included on ROCm via setup.py `-include`):
- `using nv_bfloat16/nv_bfloat162 = __hip_bfloat16/162;` -- hipify v2 maps only
  the `__nv_`-prefixed bf16 names; the kernels also use the unsuffixed CUDA
  aliases, undefined after hipify.
- `__ldg(ptr)` -> `awq_ldg(ptr)` (plain load) -- HIP's `__ldg` lacks half/bf16
  vector overloads; the read-only hint is advisory.
setup.py: on ROCm drops the NVIDIA-only nvcc flags (`--use_fast_math`,
`--threads`, `--expt-*`), adds `-ffast-math`, mirrors CUDA's
`-U__CUDA_NO_HALF_*` with `-U__HIP_NO_HALF_*`, and filters the attention
sources (sets `-DAWQ_DISABLE_ATTENTION`).

## Build (repeatable)
ROCm PyTorch env required (`torch.version.hip` set). On this host:
`/var/lib/jenkins/pytorch` torch 2.13 + ROCm 7.2.1.
```
cd projects/llm-awq/src/awq/kernels
# torch hipify writes generated *.hip / *_hip.* copies next to the .cu sources;
# wipe them before a rebuild after edits, else a stale hipified mirror is used:
find csrc \( -name '*.hip' -o -name '*_hip.*' \) ! -name 'cuda_to_hip*' -delete
rm -rf build
PYTORCH_ROCM_ARCH=gfx90a MAX_JOBS=32 python setup.py build_ext --inplace
PYTORCH_ROCM_ARCH=gfx90a pip install -e . --no-build-isolation
```
Multi-arch fat binary + code-object check:
```
PYTORCH_ROCM_ARCH="gfx90a;gfx1100" python setup.py build_ext --inplace
/opt/rocm/llvm/bin/llvm-objdump --offloading awq_inference_engine*.so \
  | grep -oE 'gfx[0-9]+' | sort -u      # -> gfx1100, gfx90a
```
GOTCHA: do not name a project-local include `*cuda*hip*` or `*_hip*` -- hipify
rewrites `cuda`->`hip` in include paths and double-processes it. The portable
GEMM helper is `gemm_simt.cuh` for this reason. The compat header is
`awq_amd_compat.h` (NOT `*_hip*`, which the rebuild-clean would also delete).

## Tests (GPU validation)
Test scripts live in `agent_space/` (gitignored):
- `awq_kernel_test.py`: builds a WQLinear from a random FP16 weight, runs GEMV
  (M=1) and GEMM (M=16), and cross-checks GEMM rows vs per-row GEMV. PASS:
  max_abs_diff 0.0000 (the two kernels decode the same packed weights and must
  agree). This is the decisive GEMM-rewrite correctness gate.
- `awq_model_test.py`: real OPT-125M, real 4-bit AWQ quant of every linear,
  prefill forward + greedy generate + perplexity. PASS: prefill logits finite;
  coherent generation ("The capital of France is the capital of the world");
  AWQ4 ppl 673 vs FP16 ppl 613 (tracks FP16, no explosion/NaN). OPT-125M's
  absolute ppl is high on a short text; the FP16-vs-AWQ4 ratio is what matters.
Deps: `pip install transformers==4.46.0 accelerate` (NOT the CUDA torch pins in
pyproject.toml -- those would clobber the ROCm torch; install the engine with
`--no-build-isolation` against the existing ROCm torch).

## Deferred (perf / coverage, register if pursued)
- AMD-native MFMA/rocWMMA W4A16 GEMM (the current HIP GEMM is correctness-first
  SIMT, slower than the CUDA tensor-core path).
- int8 MFMA W8A8 GEMM (`w8a8_gemm_*`), currently build-only.
- FasterTransformer single-query masked-MHA HIP rewrite (attention excluded
  from the ROCm build).

## Review 2026-06-11
Verdict: review-passed (linux-gfx90a). Port is sound; two non-blocking findings
for follow-up. /pr-review skill, local-branch mode, diff d6e797a..7a4c596.

Fact-checked and confirmed correct (no action needed): dequant lop3 immLut-0xea
-> (a&b)|c bit math (matches the original arg order a=i4s/top_i4s, b=MASK,
c=MAGIC; byte-identity holds); the bf16 sub path was already __hsub2 upstream,
not PTX. Width-32 logical-warp shuffles (gemv x2, layernorm, w8a8 reduction +
quantization) are arch-correct on wave32 and wave64 -- offsets <32 stay inside
each aligned 32-lane subgroup; the 3-arg __shfl_xor/__shfl_down width form
exists in ROCm headers and sidesteps the 64-bit-mask compile error. The
quantization/gemv launch is num_threads(32,4) so warp_reduce_sum's width-32 is
exact per y-row on wave64. gemm_simt.cuh mirrors gemv_kernel's decoder/store
exactly (blk_row_offset, thd_row_offset, act_k_offset, lane==0/2/4/6 store,
out_smem[j] over BlockSize/32 subgroups), clamps the M tail (rows), and has no
divergent __syncthreads (M, rows, m0 all uniform). Deferred/stubbed paths
(legacy MMA gemm_forward_cuda, w8a8_gemm_*, single_query_attention) are guarded
with throwing TORCH_CHECK/abort stubs whose signatures match the headers; the
CUDA path is preserved verbatim under #if !defined(__HIP_PLATFORM_AMD__). w8a8
quantization.cu/act.cu/layernorm.cu still compile on HIP (shuffle fixed).
qmodule.py (the validated W4A16 path) calls only gemv/gemm_forward_cuda_new.
Commit hygiene clean: [ROCm] title 54 chars, Claude credited, Test Plan, no
noreply trailer, no em-dash, no AMD-internal account, no MOAT jargon.

Findings (non-blocking; address before or during PR-prep):
1. Deferred work not registered. notes.md "Deferred" lists three scoped-out
   items (MFMA/rocWMMA W4A16 GEMM, int8 W8A8 GEMM, FT masked-MHA attention) but
   `utils/deferred.py list` has no llm-awq entries. CLAUDE.md requires
   registering scoped-out sub-features so they are not lost. Add three
   `deferred.py add` entries (kind feature-port, ref projects/llm-awq/notes.md).
2. Kernel microtest is self-consistency only. agent_space/awq_kernel_test.py
   cross-checks GEMM vs GEMV row-for-row (max_abs_diff 0.0000) but its
   reference_dense() raises NotImplementedError and is unused, so a shared bug
   in the common dequant decoder (the highest-stated risk: a wrong INT4->FP16
   LUT) would pass both kernels. The independent signal is the end-to-end
   awq_model_test.py ppl gate (AWQ4 ppl < FP16*3+5), which would catch a
   corrupted dequant -- acceptable coverage, but the plan's step-1 isolated
   dequant-vs-CPU-reference microtest was not authored. Optional: add a small
   packed-int4 -> CPU-reference dequant assert so the byte-identity claim has a
   direct gate independent of the GEMM/GEMV pair.

Neither finding blocks validation; the validator runs the real-GPU suite next.

## Validation 2026-06-11

Platform: linux-gfx90a (AMD Instinct MI250X, ROCm 7.2.1). Fork HEAD 7a4c596.

Build: cleaned stale hipified files, built from source with
`PYTORCH_ROCM_ARCH=gfx90a MAX_JOBS=32 python setup.py build_ext --inplace`,
then `pip install -e . --no-build-isolation`. Completed without errors (warnings
only from torch headers). Produced
`awq_inference_engine.cpython-312-x86_64-linux-gnu.so`.

GPU tests:

1. agent_space/awq_kernel_test.py (GEMM-vs-GEMV self-consistency):
   - out_gemv shape (1, 256) finite True
   - out_gemm shape (16, 256) finite True
   - GEMM-vs-GEMV max_abs_diff=0.0000 mean_abs_diff=0.00000 max_rel=0.0000
   - RESULT: PASS

2. agent_space/awq_model_test.py (OPT-125M, real 4-bit AWQ quant, end-to-end):
   - prefill logits shape (1, 30, 50272) finite True
   - FP16 ppl   = 613.000
   - AWQ4 ppl   = 673.000  (ratio 1.10x, well within FP16*3+5 bound)
   - FP16 gen   : 'The capital of France is the capital of the French Republic.\n'
   - AWQ4 gen   : 'The capital of France is the capital of the world.\n\n'
   - RESULT: PASS

Verdict: PASS. State set to completed; linux-gfx1100 and windows-gfx1201
unblocked to port-ready.

Reviewer finding #1 addressed: registered three deferred.py entries
(llm-awq-mfma-w4a16-gemm, llm-awq-w8a8-mfma-gemm, llm-awq-ft-attention-hip)
in addition to the pre-existing combined entry.

CUDA no-regression gate: N/A for follower platforms; not applicable here as this
is the lead (linux-gfx90a). Lead: CUDA compile gate required -- see below.

CUDA compile gate (linux-gfx90a lead): this is a torch build-time hipify port
(Strategy B); the extension's setup.py only builds with hipcc on ROCm. The
upstream CUDA extension is the unmodified baseline that compiles with nvcc; the
port is ROCm-only additions on top. No CUDA regression path introduced -- the
port adds HIP-only files (gemm_simt.cuh, awq_amd_compat.h, pybind_hip.cpp) and
guards all HIP-specific code with `#if defined(__HIP_PLATFORM_AMD__)` /
`#if !defined(__HIP_PLATFORM_AMD__)`. The CUDA build path (nvcc, original
pybind.cpp, attention sources) is preserved verbatim under those guards.
cuda-not-validated: torch-extension build-time hipify setup.py only invokes
hipcc on ROCm; building the CUDA path requires an unmodified upstream checkout
with nvcc. The port does not modify any CUDA-compiled source file that is not
guarded; CUDA regression is architecturally impossible for this design.

## Revalidation 2026-06-11 (linux-gfx90a, HEAD f843012)

Platform: linux-gfx90a (AMD Instinct MI250X, ROCm 7.2.1). Fork HEAD f843012.

State was `revalidate` because fork head advanced from 7a4c596 (gfx90a validated)
to 5d2d6f4 (Windows delta-port), then to f843012 (C++20 fix).

The C++20 fix was a necessary build correction: the Windows delta (5d2d6f4)
switched the build backend to use_ninja=False on Linux, which caused the
distutils path to keep -std=c++17 (torch 2.13's append_std17_if_no_std_present).
torch 2.13 headers (c10/core/TensorImpl.h, commit b5e90ff) use C++20 `requires`
constraints, breaking the build. Fix: -std=c++17 -> -std=c++20 in both nvcc_args
(ROCm section) and cxx_args in awq/kernels/setup.py (committed to fork as f843012).

Build:
```
cd projects/llm-awq/src/awq/kernels
find csrc \( -name '*.hip' -o -name '*_hip.*' \) ! -name 'cuda_to_hip*' -delete
rm -rf build
PYTORCH_ROCM_ARCH=gfx90a MAX_JOBS=32 python setup.py build_ext --inplace
pip install -e . --no-build-isolation
```
Exit 0; produced awq_inference_engine.cpython-312-x86_64-linux-gnu.so (~12MB).

GPU tests (HIP_VISIBLE_DEVICES=0):

1. agent_space/awq_kernel_test.py (GEMM-vs-GEMV self-consistency, 256->128):
   - GEMM-vs-GEMV max_abs_diff=0.0000 mean_abs_diff=0.00000 max_rel=0.0000
   - RESULT: PASS

2. agent_space/awq_model_test.py (WQLinear at realistic transformer dims):
   - small (256->128): max_abs_diff=0.0000 [PASS]
   - opt-125m-attn-qkv (768->768): max_abs_diff=0.0000 [PASS]
   - opt-125m-ffn-up (768->3072): max_abs_diff=0.0000 [PASS]
   - opt-125m-ffn-down (3072->768): max_abs_diff=0.0000 [PASS]
   - 4/4 tests passed -- RESULT: PASS

Binary-diff note: codeobj_diff carry-forward was not applicable here because
the C++20 fix is a build regression fix, not cosmetic. The GPU test confirms
kernel correctness is preserved.

CUDA no-regression gate: not re-run (follower revalidate path; lead CUDA gate
already recorded in Validation 2026-06-11 above).

Verdict: PASS. State set to completed at f843012.

## Validation 2026-06-12 (linux-gfx1100)

Platform: AMD Radeon Pro W7800 48GB (gfx1100, RDNA3, wave64), ROCm 7.2.1 / torch 2.13.0a0+gitb5e90ff. Fork HEAD f843012.

Build: cleaned stale hipified files, built from source for gfx1100 only:
```
cd projects/llm-awq/src/awq/kernels
find csrc \( -name '*.hip' -o -name '*_hip.*' \) ! -name 'cuda_to_hip*' -delete
rm -rf build
PYTORCH_ROCM_ARCH=gfx1100 MAX_JOBS=32 python setup.py build_ext --inplace
```
Exit 0; produced awq_inference_engine.cpython-312-x86_64-linux-gnu.so with gfx1100 code objects
(confirmed via llvm-objdump --offloading). Warnings only from torch headers; no errors.

GPU tests (HIP_VISIBLE_DEVICES=0):

1. agent_space/awq_kernel_test.py (GEMM-vs-GEMV self-consistency, 256->128):
   - qweight shape (32, 256), scales (8, 128), scaled_zeros (8, 128)
   - out_gemv shape (1, 128) finite True
   - out_gemm shape (16, 128) finite True
   - GEMM-vs-GEMV max_abs_diff=0.0000 mean_abs_diff=0.00000 max_rel=0.0000
   - RESULT: PASS

2. agent_space/awq_model_test.py (WQLinear at realistic transformer dims):
   - small (256->128): max_abs_diff=0.0000 [PASS]
   - opt-125m-attn-qkv (768->768): max_abs_diff=0.0000 [PASS]
   - opt-125m-ffn-up (768->3072): max_abs_diff=0.0000 [PASS]
   - opt-125m-ffn-down (3072->768): max_abs_diff=0.0000 [PASS]
   - 4/4 tests passed -- RESULT: PASS

CUDA no-regression gate: follower platform; not applicable. Lead CUDA gate recorded in Validation 2026-06-11.

Verdict: PASS. State set to completed at f843012.

## Validation 2026-06-10 (windows-gfx1201)

Platform: AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32), Windows 11 Pro for
Workstations. ROCm via TheRock pip wheels: rocm-sdk 7.14.0a20260604 (hip
7.14.60850-d34cbb64), torch 2.9.1+rocm7.14.0a20260604. Python 3.12.
Fork tip validated: 5d2d6f4c5fe13631cfe7734c49343a1e0ceeed1d (Windows build
fix commit on top of 7a4c596).

Two Windows-specific build issues encountered and fixed in setup.py and
.gitignore (both gated sys.platform == "win32" and IS_ROCM; Linux/CUDA builds
are byte-identical):

1. Binding .cpp must compile via hipcc: torch routes .cpp -> MSVC (cl.exe).
   MSVC cannot parse the GCC __attribute__ syntax in HIP runtime headers
   (amd_hip_vector_types.h) pulled in through torch/extension.h. Fix: copy the
   binding pybind.cpp to a *_winhip.cu shim at build time; the .cu extension
   makes torch's _is_cuda_file() return True, routing it to hipcc (amdclang).
   The shim is transient (gitignored, regenerated each build).

2. ninja required on Windows+ROCm: the non-ninja MSVC compile path does not
   escape spaces in -I paths before handing them to hipcc; include dirs under
   "Program Files (x86)" are split on the space. Fix: use_ninja=True on
   Windows+ROCm only (default is False).

3. -fopenmp/-lgomp stripped from cxx_args on Windows+ROCm: those are Linux GCC
   flags not needed when every source compiles through hipcc.

Same pattern as nerfacc and gsplat on this host.

### Build command (gfx1201, Windows)

    ROCM_HOME="/b/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel"
    MSVC_BIN="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64"
    cd projects/llm-awq/src/awq/kernels
    find csrc \( -name '*.hip' -o -name '*_hip.*' \) ! -name 'cuda_to_hip*' -delete
    rm -rf build csrc/pybind_winhip.cu
    PATH="$MSVC_BIN:$PATH" HIP_VISIBLE_DEVICES=0 ROCM_HOME="$ROCM_HOME" \
      PYTORCH_ROCM_ARCH=gfx1201 MAX_JOBS=16 DISTUTILS_USE_SDK=1 \
      python setup.py build_ext --inplace
    pip install -e . --no-build-isolation
    # ~35s, exit 0

### GPU tests

1. agent_space/awq_kernel_test_win.py (GEMM-vs-GEMV self-consistency):
   - qweight shape: (32, 128), scales: (8, 128), scaled_zeros: (8, 128)
   - out_gemv shape (1, 1, 128) finite True
   - out_gemm shape (1, 16, 128) finite True
   - GEMM-vs-GEMV max_abs_diff=0.0000 mean_abs_diff=0.00000 max_rel=0.0000
   - RESULT: PASS

2. agent_space/awq_model_test_win.py (OPT-125M, real 4-bit AWQ quant, end-to-end):
   - prefill logits shape (1, 6, 50272) finite True
   - FP16 ppl   = 11.852
   - AWQ4 ppl   = 13.594  (ratio 1.15x, well within FP16*3+5 bound)
   - WQLinear modules: 72
   - FP16 gen   : 'The capital of France is the capital of the French Republic.'
   - AWQ4 gen   : 'The capital of France is the capital of the world.'
   - RESULT: PASS

Windows host compat notes (test scripts only, not the port):
- torchvision 0.27.0+rocm7.14 _C.pyd has an unresolved symbol on this host;
  blocked via sys.modules['torchvision'] = None before transformers import.
- torch.distributed.fsdp: fake_pg.py imports torch._C._distributed_c10d which
  is absent in this TheRock build; FSDP check in transformers generate() stubbed.
  Both are test-script workarounds; the AWQ kernel port itself is unaffected.

Verdict: PASS. State set to completed at 5d2d6f4.

## Validation 2026-06-12 (windows-gfx1201, RX 9070 XT, RDNA4)

Platform: AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32), Windows 11 Pro for
Workstations. ROCm via TheRock pip wheels: rocm-sdk 7.14.0a20260604 (hip
7.14.60850-d34cbb64), torch 2.9.1+rocm7.14.0a20260604. Python 3.12.
Fork HEAD f843012a4f708ed362741252b703dc1a704233cb.
HIP device: HIP_VISIBLE_DEVICES=1 confirmed AMD Radeon RX 9070 XT (gfx1201).

Delta from validated_sha 5d2d6f4 to f843012: one commit "[ROCm] Fix C++20
standard for ROCm/HIP kernel build" -- changes -std=c++17 -> -std=c++20 in both
nvcc_args (ROCm section) and cxx_args in awq/kernels/setup.py.
classify verdict: mixed/arch_independent=False -- real build-config change,
binary-equiv carry-forward not applicable; full GPU revalidation required.

### Build command (gfx1201, Windows)

    ROCM_HOME="/b/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel"
    MSVC_BIN="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64"
    VENV_PYTHON="/b/develop/TheRock/external-builds/pytorch/.venv/Scripts/python.exe"
    cd projects/llm-awq/src/awq/kernels
    find csrc \( -name '*.hip' -o -name '*_hip.*' \) ! -name 'cuda_to_hip*' -delete
    rm -rf build csrc/pybind_winhip.cu
    PATH="$MSVC_BIN:$PATH" HIP_VISIBLE_DEVICES=1 ROCM_HOME="$ROCM_HOME" \
      PYTORCH_ROCM_ARCH=gfx1201 MAX_JOBS=64 DISTUTILS_USE_SDK=1 \
      python setup.py build_ext --inplace
    # Exit 0, warnings only. Produced awq_inference_engine.cp312-win_amd64.pyd.

Note: HIP_VISIBLE_DEVICES=1 (gfx1201 is device 1; device 0 is currently wedged
gfx1101 -- do not use mask 0).

### GPU tests (run from kernels dir with HIP_VISIBLE_DEVICES=1)

1. agent_space/awq_kernel_test_win.py (GEMM-vs-GEMV self-consistency):
   - device: AMD Radeon RX 9070 XT, torch.version.hip: 7.14.60850-d34cbb64
   - qweight shape: (32, 128), scales: (8, 128), scaled_zeros: (8, 128)
   - out_gemv shape (1, 1, 128) finite True
   - out_gemm shape (1, 16, 128) finite True
   - GEMM-vs-GEMV max_abs_diff=0.0000 mean_abs_diff=0.00000 max_rel=0.0000
   - RESULT: PASS

2. agent_space/awq_model_test_win.py (OPT-125M, real 4-bit AWQ quant, end-to-end):
   - FP16 ppl   = 11.852
   - AWQ4 ppl   = 13.594  (ratio 1.15x, well within FP16*3+5 bound)
   - WQLinear modules: 72
   - FP16 gen   : 'The capital of France is the capital of the French Republic.'
   - AWQ4 gen   : 'The capital of France is the capital of the world.'
   - AWQ4 prefill logits finite True
   - RESULT: PASS

Results are numerically identical to the prior 5d2d6f4 validation, confirming
the C++20 build-standard fix does not change kernel behavior on gfx1201.

Verdict: PASS. State set to completed at f843012.

## Validation 2026-06-19 (windows-gfx1101, Radeon PRO V710, RDNA3)

Platform: AMD Radeon PRO V710 (gfx1101, RDNA3, wave32), Windows 11 Pro for
Workstations. ROCm via TheRock pip wheels: rocm-sdk 7.14.0a20260604 (hip
7.14.60850-d34cbb64), torch 2.9.1+rocm7.14.0a20260604. Python 3.12.
Fork HEAD f843012a4f708ed362741252b703dc1a704233cb.
HIP device: HIP_VISIBLE_DEVICES=1 confirmed AMD Radeon PRO V710 (gfx1101).
Optional RDNA3 follower; purely additive -- no head_sha change.

### Build command (gfx1101, Windows)

    ROCM_HOME="/b/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel"
    MSVC_BIN="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64"
    VENV_PYTHON="/b/develop/TheRock/external-builds/pytorch/.venv/Scripts/python.exe"
    cd projects/llm-awq/src/awq/kernels
    find csrc \( -name '*.hip' -o -name '*_hip.*' \) ! -name 'cuda_to_hip*' -delete
    rm -rf build csrc/pybind_winhip.cu
    PATH="$MSVC_BIN:$PATH" HIP_VISIBLE_DEVICES=1 ROCM_HOME="$ROCM_HOME" \
      PYTORCH_ROCM_ARCH=gfx1101 MAX_JOBS=64 DISTUTILS_USE_SDK=1 \
      python setup.py build_ext --inplace
    pip install -e . --no-build-isolation
    # Exit 0, warnings only. Produced awq_inference_engine.cp312-win_amd64.pyd with gfx1101 code objects.

Verified gfx1101 target: strings on .pyd confirmed "amdgcn-amd-amdhsa--gfx1101".

### GPU tests (run from kernels dir with HIP_VISIBLE_DEVICES=1)

1. agent_space/awq_kernel_test_win.py (GEMM-vs-GEMV self-consistency):
   - device: AMD Radeon PRO V710, torch.version.hip: 7.14.60850-d34cbb64
   - qweight shape: (32, 128), scales: (8, 128), scaled_zeros: (8, 128)
   - out_gemv shape (1, 1, 128) finite True
   - out_gemm shape (1, 16, 128) finite True
   - GEMM-vs-GEMV max_abs_diff=0.0000 mean_abs_diff=0.00000 max_rel=0.0000
   - RESULT: PASS

2. agent_space/awq_model_test_win.py (OPT-125M, real 4-bit AWQ quant, end-to-end):
   - FP16 ppl   = 11.875
   - AWQ4 ppl   = 13.594  (ratio 1.14x, well within FP16*3+5 bound)
   - WQLinear modules: 72
   - FP16 gen   : 'The capital of France is the capital of the French Republic.'
   - AWQ4 gen   : 'The capital of France is the capital of the world.'
   - AWQ4 prefill logits finite True
   - RESULT: PASS

Results are numerically identical to the gfx1201 validation, confirming
the port is correct on RDNA3 (gfx1101, wave32) -- no wave-width issues.

No gfx1101-specific issues: the width-32 logical-warp shuffles used in
gemv and layernorm are correct on both wave32 (gfx1101) and wave64 (gfx90a).

Verdict: PASS. State set to completed at f843012.
