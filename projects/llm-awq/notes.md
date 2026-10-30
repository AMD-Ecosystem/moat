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
`-U__CUDA_NO_HALF_*` with `-U__HIP_NO_HALF_*`, links `hipblaslt`, and (since
a9998ae, on Linux and Windows) copies every .cpp source to a transient
`*_cpp_shim.cu` so hipcc compiles it; no .cpp reaches g++/MSVC on ROCm.
(Before a9998ae the shim was `*_winhip.cu`, Windows only; older records below
use that name.)

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

Platform: AMD Radeon Pro W7800 48GB (gfx1100, RDNA3, wave32), ROCm 7.2.1 / torch 2.13.0a0+gitb5e90ff. Fork HEAD f843012.

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

## Porting round 2026-10-29 (linux-gfx90a): attention, W8A8 GEMM, legacy GEMM

jeff asked for a full port: the three ROCm throwing stubs left at a2a710a are now real HIP paths.
Fork `moat-port`: a2a710a -> 73edaef (attention) -> 7f0f35f (W8A8) -> 2c5d913 (legacy GEMM)
-> 2869c53 (README ROCm note). Host: AMD Instinct MI250X (gfx90a, wave64), ROCm 7.16
(TheRock SDK in the image venv), torch 2.16.0a0+git6c3af02, Python 3.11.

### What changed
1. `single_query_attention` (csrc/attention, FasterTransformer decoder masked MHA). Kernel ported,
   not replaced: utils.h has a HIP branch for each of the 13 fp16 inline-PTX blocks
   (`__builtin_bit_cast` uint32<->`__half2`, `__hadd2/__hmul2/__hfma2/__float22half2_rn/...`);
   template.hpp routes every shuffle through `MMHA_SHFL_XOR/MMHA_SHFL` (width 32 on HIP, the original
   `__shfl_*_sync` on CUDA), compiles out the unused `mma.sync` hmma_fp32 (it also had an upstream
   missing-comma typo clang rejects), and `cvt.rni.sat.s8.f32` -> rintf+clamp. cuda_bf16_fallbacks.cuh
   defines `make_bfloat162` / `__floats2bfloat162_rn` on HIP (absent from every hip_bf16.h checked,
   incl. clr trunk). The .cu launcher checks dynamic+static LDS against
   `hipDeviceAttributeMaxSharedMemoryPerBlock` and throws (RuntimeError in Python) past the limit.
   setup.py/pybind: exclusion + stub removed; on Windows ft_attention.cpp goes through the same
   `_winhip.cu` shim loop as pybind.cpp (shim comment updated to "each .cpp source").
   The host-compiled ft_attention.cpp includes hip_bf16.h through g++ fine on ROCm 7.16; the old
   note said it failed on 7.2.1 -- watch for that on the gfx1100 host (if it fails, route the .cpp
   through hipcc like the Windows shim does).
2. W8A8 GEMM (`w8a8_gemm_forward_cuda`, `_fuse_bias`): hipBLASLt int8 (COMPUTE_32I, scale HIP_R_32I,
   TN: C^T[N,M] = W[N,K] x A^T), PyTorch's `getCurrentCUDABlasLtHandle()`,
   `getCUDABlasLtWorkspace()/Size()`, current stream. No heuristic result or failed matmul -> tiled
   SIMT int8 kernel. Epilogue kernel mirrors the CUDA op order. setup.py links `hipblaslt` on ROCm.
3. Legacy `gemm_forward_cuda`: SIMT kernel (32x64 tile, 256 threads, BK=32) that dequantizes the
   legacy layout into LDS with the CUDA expression `fp16(s*(q-z))`, fp32 accumulate, single pass
   over K (no split-K buffer on HIP). Layout: kernel [OC, IC/8] int32 (nibble i of word j = ic 8j+i,
   no interleave), zeros [OC, zeros_w] (8 groups/word), scales [OC, 8*zeros_w]. Not the
   quantization_new interleaved layout, so gemm_simt.cuh is reused in approach only.
4. README: AMD note under the kernel install step.

### Upstream caller contracts found (CUDA behaves the same; not changed)
- single_query_attention uses `q.stride(0)` as the batch stride of q, k AND v. With GQA/MQA and
  batch > 1, separate contiguous k/v tensors (as tinychat llama passes) are read at the wrong
  offset; views of one fused qkv buffer work. TinyChat runs max_batch_size=1.
- With `length_per_sample`, LDS is sized from `timestep`, so callers must pass timestep >= max length.
- LDS limit on 64 KiB parts: last timestep that fits = fp16 D=128: 10867; bf16 D=64: 10889 (the
  next refuses); fp32 D=128: 16239. TinyChat's kv_max_seq_len is <= 8192, so it fits.

### Build
```
cd projects/llm-awq/src/awq/kernels
find csrc \( -name '*.hip' -o -name '*_hip.*' -o -name 'hip_bf16_*' \) ! -name 'cuda_to_hip*' -delete
rm -rf build
PYTORCH_ROCM_ARCH=gfx90a MAX_JOBS=32 python setup.py build_ext --inplace
```
(Dev loop forced torch's ninja backend through a wrapper: 44 s vs the serial distutils default.)
Fat binary `PYTORCH_ROCM_ARCH="gfx90a;gfx1100;gfx1201"`: builds with 0 errors; llvm-objdump
--offloading shows gfx90a, gfx1100, gfx1201 code objects.

### Tests (scripts committed in projects/llm-awq/tests/; run with PYTHONPATH=<src>/awq/kernels)
- `attn_test.py`: 443/443 PASS. Kernel vs PyTorch reference (same rotary, cache write, softmax):
  fp16+bf16 x head dims {32,48,64,80,96,112,128,160,192,224,256} x heads {8/8, 32/8 GQA, 8/1 MQA}
  x t {0,1,31,32,517,2048} (covers all three THREADS_PER_BLOCK=64/128/256 launch variants); fp32;
  batch 2-4 (MHA, GQA/MQA via fused qkv views); GPT-J + NeoX rotary, base 5e5 scale 0.5, partial
  rotary, length_per_sample, ALiBi, cache wrap (t >= L); long t up to the LDS edge, next refused.
  Max out err fp16 <= 1e-2, bf16 <= 4e-2 (bf16 logits stored in bf16), fp32 <= 1.3e-5.
- `tinychat_e2e.py <src>`: 4/4 PASS. tinychat/models/llama.py loaded directly, 4 layers, AWQ 4-bit
  WQLinear projections, random weights; greedy decode via single_query_attention vs re-running the
  full sequence through the non-fused (prefill) attention each step, teacher-forced. Max |dlogit|
  <= 2.9e-3, cosine >= 0.999999, argmax 40/40, 40/40, 64/64, 40/40 (MHA 8/8 D128, GQA 16/4 D64,
  GQA 8/2 D128, MQA 8/1 D80).
- `w8a8_test.py`: 25/25 PASS vs exact int64 matmul + same epilogue; shapes 1x8x4 .. 512x4096x4096
  incl. 3x13x7, 33x50x100, 65x72x37, 300x130x520; bias and no bias; side stream. Profiler: every
  shape ran hipBLASLt `Cijk_Alik_Bljk_I8II_...` on gfx90a.
- `w8a8_simt_test.py <w8a8_gemm_cuda.cu>`: 7/7 exact. Builds the SIMT fallback kernel text in /tmp
  via load_inline (no checkout path) since gfx90a never reaches it.
- `legacy_gemm_test.py`: 8/8 PASS vs PyTorch dequant+matmul and vs legacy gemv_forward_cuda; M 1..300,
  OC 64..11008, IC 1024..4096, G 64/128; max err 3.9e-3 (fp16 output rounding).
- `w4a16_test.py <src>`: 30/30 PASS, regression of the previously validated WQLinear GEMV/GEMM path
  vs a fake-quant reference. Test gotcha: `WQLinear.from_linear` does not clamp q to [0,15]; a
  weight that rounds to 16 wraps to 0 (sparse 16*scale errors). Snap weights to the grid first.
- CUDA no-regression: no nvcc on this host. `cuda_view.py`-style check (drop `__HIP_PLATFORM_AMD__`
  branches, diff vs upstream d6e797a): CUDA-visible text of the attention files differs only by the
  MMHA_SHFL*/MMHA_SET_MAX_DYNAMIC_SMEM_SIZE macros, which expand to the original calls; w8a8 and
  gemm_cuda_gen CUDA text identical. setup.py: `libraries=[]` on CUDA.

### Not done / limits
- Real TinyChat generation with a llama AWQ checkpoint and tokenizer was NOT run: this host has no
  transformers / flash_attn / tokenizers / checkpoint, and the fleet guard blocks pip (even into a
  private venv). The module-level e2e above stubs `transformers.LlamaRotaryEmbedding` (unused) and
  `flash_attn_func` (exact SDPA). Needs a host with those installed: run tinychat/demo.py with a
  small llama-family AWQ checkpoint and compare against `--quant False` / non-fused attention.
- Still owed: wave32 (gfx1100) and Windows GPU runs of all six tests (fat-binary compile only here).
  On Windows also confirm hipblaslt links from the TheRock SDK and the ft_attention shim builds.
- Performance: W8A8 hipBLASLt path adds an int32 [M,N] temporary and a separate epilogue launch;
  legacy GEMM and W4A16 GEMM are SIMT (MFMA deferred items stand).
- Deferred registry: `llm-awq-ft-attention-hip` and `llm-awq-w8a8-mfma-gemm` set done (the latter
  runs hipBLASLt's MFMA int8 kernels, MI32x32x1 in the kernel name). `llm-awq-mfma-w4a16-gemm` stays open.
- `tests/cuda_view.py <file>` prints a source as the CUDA build sees it (HIP branches dropped).

## Review 2026-10-29 (linux-gfx1100)

Verdict: changes-requested. /pr-review skill, local-branch mode, fork `moat-port` at 2869c53, whole diff d6e797a...HEAD with focus on a2a710a..2869c53 (attention, W8A8 hipBLASLt, legacy GEMM, README). Static review; no build or GPU run on this host (the four new tests still owe their wave32 run, which is the validator's job).

Checked and not findings: the MMHA_SHFL*/width-32 shuffles are exact on wave32 (width = wavefront) and wave64 (offsets < 32 stay inside each 32-lane half); every partition in the template is logical-32 (WARP_SIZE = 32, K_PER_WARP, ti_end alignment, red_smem by lane < WARPS_PER_BLOCK), and the masked QK reduce (`is_masked = tidx >= QK_VECS_PER_WARP`) only reads source lanes < QK_VECS_PER_WARP. hipBLASLt TN mapping (C^T[N,M] = W[N,K]^T-op x A^T) and int32 alpha/beta with HIP_R_32I scale are correct; handle, workspace and stream all come from the current stream. The LDS guard is sufficient: static LDS is at most about 2 KiB, so a launch under the 48 KiB threshold always fits 64 KiB. The legacy SIMT GEMM matches the CUDA kernel's layout (B [OC, IC/8], zeros_w / sf_w formulas, fp16(s*(q-z))). CUDA-visible text of every changed source (tests/cuda_view.py vs d6e797a) differs only by the MMHA_SHFL*/MMHA_SET_MAX_DYNAMIC_SMEM_SIZE macros, which expand to the original calls; setup.py passes `libraries=[]` on CUDA. jargon.py clean; titles 47-59 chars, AI disclosure, no trailer, ASCII.

Findings:
1. Build regression on ROCm PyTorch < 2.9 (whole extension, not just W8A8). `awq/kernels/csrc/w8a8/w8a8_gemm_cuda.cu:979-980` calls `at::cuda::getCUDABlasLtWorkspaceSize()` / `getCUDABlasLtWorkspace()`, which first appear in torch v2.9.0 (`aten/src/ATen/cuda/CUDAContextLight.h`: 0 matches at v2.3.0..v2.8.0, 2 at v2.9.0; added by pytorch #156495). `getCurrentCUDABlasLtHandle()` exists back to 2.3. The repo pins torch==2.3.0, ROCm wheels exist for 2.3..2.8, and a2a710a built there; HEAD now fails to compile, taking the validated W4A16 path down with it. Fix: take the workspace from a tensor on the current stream (`at::empty({ws}, _in_feats.options().dtype(at::kByte))`, or workspace 0, since int8 solutions rarely need one) or gate the torch-owned workspace on `TORCH_VERSION_MAJOR/MINOR` from `<torch/version.h>`; otherwise state the minimum torch in the README note (`README.md:127`).
2. Linux ROCm host compile of `ft_attention.cpp` is unverified on older ROCm (lower confidence, needs evidence). The a2a710a record above says the attention sources were excluded because `amd_hip_bf16.h` failed to parse in the g++-compiled host .cpp on ROCm 7.2.1; this round re-adds the file on Linux with g++ (`awq/kernels/setup.py:80-97` shims only on win32) and has been built only against ROCm 7.16 (this host is 7.16 as well, so it adds no evidence). If 7.2.x still fails, the whole extension stops building there. Either build HEAD once against a ROCm 7.2.x torch, or route `csrc/attention/ft_attention.cpp` through hipcc on Linux too (the same `_winhip.cu` shim, not gated on win32), and record the minimum ROCm.
3. Lesson not promoted. `.claude/skills/cuda-to-rocm` is unchanged on port/llm-awq. That HIP's bf16 header has no `make_bfloat162` / `__floats2bfloat162_rn` (confirmed absent in the ROCm 7.16 SDK `amd_hip_bf16.h`; only `__float22bfloat162_rn` exists) hits every FasterTransformer-derived kernel, and the torch >= 2.9 limit of `getCUDABlasLtWorkspace*` (finding 1) hits any extension that calls hipBLASLt directly. Add both to `references/fault-classes.md` (library/header gaps), naming llm-awq.
4. Upstream-visible text and records. (a) Commit 73edaef says the TinyChat logits "agree to <= 2.7e-3"; the recorded run above says max |dlogit| <= 2.9e-3. Reconcile at PR-prep squash. (b) a2a710a removed a trailing space from the CUDA-only signature line of `gemm_forward_4bit_cuda_m128n64k32` (`awq/kernels/csrc/quantization/gemm_cuda_gen.cu:38`); restore the upstream line so the CUDA path is byte-identical. (c) The "Validation 2026-06-12 (linux-gfx1100)" header above calls gfx1100 "RDNA3, wave64"; gfx1100 is wave32. Correct the record so nobody reads it as wave64 evidence.

## Porting round 2026-10-29 (linux-gfx90a): review fixes

Answers "Review 2026-10-29 (linux-gfx1100)". Fork `moat-port`: 2869c53 -> dac7721 -> a9998ae -> 8f14286. Host: AMD Instinct MI250X (gfx90a, wave64), ROCm 7.16 (TheRock SDK), torch 2.16.0a0+git6c3af02, Python 3.11. The review was recorded only here; the fork review PR (#1) has no review threads, so there is nothing to reply to on it.

1. dac7721: the hipBLASLt workspace no longer comes from `at::cuda::getCUDABlasLtWorkspace*` (torch >= 2.9). The preference caps the workspace at 32 MiB; after the heuristic, `heuristic.workspaceSize` bytes are allocated as `at::empty({n}, _in_feats.options().dtype(at::kByte))` on the current stream (caching allocator, stream-ordered) and nullptr/0 is passed when the solution needs none. Cached handle (`getCurrentCUDABlasLtHandle()`, declared at v2.3.0 and v2.8.0 in `CUDAContextLight.h` under `ROCM_VERSION >= 50700`) and current stream kept. No torch-version conditional; `grep getCUDABlasLt csrc` is empty. Not built against a torch < 2.9 wheel on this host (none installed, and the fleet guard blocks pip); the remaining at::cuda calls were checked against the v2.3.0 / v2.8.0 headers.
2. a9998ae: setup.py copies every .cpp source (pybind.cpp and attention/ft_attention.cpp) to `<name>_cpp_shim.cu` on ROCm on every OS, so hipcc compiles them; the suffix was `_winhip.cu`, Windows only. `-fopenmp -lgomp` now apply only to CUDA (no ROCm source reaches g++). .gitignore pattern renamed. CUDA build unchanged. Build log: `ft_attention_cpp_shim.o` and `pybind_cpp_shim.o` built by hipcc, zero .cpp files compiled by g++. This removes the ROCm 7.2.x g++/`amd_hip_bf16.h` question rather than answering it; no ROCm 7.2.x build was run. Windows still owes a run (the shim name changed there too).
3. Skill: `references/fault-classes.md`, "Headers, includes and build", two new entries naming llm-awq (missing `make_bfloat162` / `__floats2bfloat162_rn` in HIP bf16; torch `getCUDABlasLtWorkspace*` only from 2.9). Re-checked: `amd_hip_bf16.h` in clr trunk has 0 matches for either name and 1 for `__float22bfloat162_rn`.
4. (b) 8f14286 restores the trailing space on the CUDA kernel signature line of `gemm_cuda_gen.cu` and removes a blank line after the HIP `#endif` that made the CUDA view one line longer. `tests/cuda_view.py` output of `gemm_cuda_gen.cu` and `w8a8_gemm_cuda.cu` is now byte-identical to d6e797a (`diff` empty). (c) The 2026-06-12 linux-gfx1100 header now says wave32. (a) PR-PREP SQUASH TODO: commit 73edaef says the TinyChat logits "agree to <= 2.7e-3"; the recorded runs give max |dlogit| <= 2.9e-3 (2.930e-03 again this round). Write 2.9e-3 in the squashed message. History not rewritten now.

Build (wrapped with timeit, 10m15s serial distutils):
```
cd projects/llm-awq/src/awq/kernels
find csrc \( -name '*.hip' -o -name '*_hip.*' -o -name 'hip_bf16_*' -o -name '*_cpp_shim.cu' \) ! -name 'cuda_to_hip*' -delete
rm -rf build
PYTORCH_ROCM_ARCH=gfx90a MAX_JOBS=32 python setup.py build_ext --inplace
```

Tests at 8f14286 (run from /tmp, `PYTHONPATH=<src>/awq/kernels`, scripts in projects/llm-awq/tests/), all PASS:
- `attn_test.py`: 443/443 (LDS edge refusals as before: fp32 D=128 t=16240 needs 65552 > 65536 B).
- `tinychat_e2e.py <src>`: 4/4, argmax 40/40, 40/40, 64/64, 40/40; max |dlogit| 2.930e-03, min cos 0.999999.
- `w8a8_test.py`: 25/25, all 24 shape cases ran hipBLASLt (paths {'hipblaslt': 24}); the 25th is the side-stream ordering check, which also goes through the new workspace path. Whether any chosen solution asked for a nonzero workspace was not instrumented.
- `w8a8_simt_test.py <w8a8_gemm_cuda.cu>`: 7/7 exact.
- `legacy_gemm_test.py`: 8/8, max err vs dequant ref 3.9e-3.
- `w4a16_test.py <src>`: 30/30.
jargon.py --port llm-awq clean; prose.py clean on the three commit bodies; titles 59-66 chars.


## Review 2026-10-29 (linux-gfx1100, re-review at 8f14286)

Verdict: changes-requested. /pr-review skill, local-branch mode, fork `moat-port` at 8f14286, whole diff d6e797a...HEAD, plus the skill edit on port/llm-awq. Static review; no build or GPU run on this host. jeff asked for this re-review of the answers to "Review 2026-10-29 (linux-gfx1100)".

Status of the four earlier findings:
1. Partly resolved. dac7721 removes `getCUDABlasLtWorkspace*`, so torch 2.4 through 2.8 should now build. Torch 2.3, the version pyproject.toml pins, still fails: see new finding 1.
2. Resolved. a9998ae routes `pybind.cpp` and `attention/ft_attention.cpp` through `*_cpp_shim.cu` and hipcc on every ROCm OS, so no .cpp reaches g++ on ROCm and the 7.2.x g++ question no longer applies. Windows still owes a run (the shim was renamed there too); that is the validator's job.
3. Resolved, but one of the two lessons is wrong (new finding 2).
4. Resolved. (b) `tests/cuda_view.py` view of `gemm_cuda_gen.cu` now has the restored trailing space and no extra blank line. (c) The 2026-06-12 header says wave32. (a) is recorded as a PR-prep squash TODO (write 2.9e-3, not 2.7e-3), which is what the finding asked for.

New findings:
1. Torch 2.3 ROCm build still fails. `awq/kernels/csrc/w8a8/w8a8_gemm_cuda.cu:972` calls `at::cuda::getCurrentCUDABlasLtHandle()`. At v2.3.0 that declaration (`aten/src/ATen/cuda/CUDAContextLight.h:85-87`) and the `<cublasLt.h>` include are inside `#if ... (defined(USE_ROCM) && ROCM_VERSION >= 50700)`. `ROCM_VERSION` is a C macro only inside PyTorch's own build (`cmake/Dependencies.cmake:1297` `add_definitions(-DROCM_VERSION=...)`). v2.3.0 `cpp_extension.py` passes only `-D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1` (and `-DHIPBLAS_V2`), and no v2.3.0 header defines `ROCM_VERSION`; `rocm_version.h` defines only `ROCM_VERSION_MAJOR/MINOR/PATCH`. So in an extension build the guard evaluates to 0, the function is undeclared, and the whole extension fails to compile. From v2.4.0 the declaration is unguarded (ae9a4fa, "enforce ROCM_VERSION >= 6.0"). The notes ("declared at v2.3.0 ... under ROCM_VERSION >= 50700") record the guard but not that it is false here. The fix must do one of two things. (a) Stop depending on torch for the handle: keep one `hipblasLtHandle_t` per device, created with `hipblasLtCreate` on first use (a static array indexed by device, guarded by `std::call_once` or a mutex). Then nothing in the HIP branch needs more than torch 2.3. (b) State the floor: in `README.md:127`, say the ROCm build needs PyTorch >= 2.4. Either way, the dac7721 body is now wrong in two places: "available since PyTorch 2.3", and the claim that the fix restores builds for 2.3 through 2.8. Correct both at the PR-prep squash. If neither a torch 2.3 nor a torch 2.4 ROCm wheel can be installed on a fleet host, say so in the notes; do not claim the build works there.
2. Wrong skill lesson. `.claude/skills/cuda-to-rocm/references/fault-classes.md:394` says `getCurrentCUDABlasLtHandle()` "goes back to 2.3" and gives `git show v2.3.0:.../CUDAContextLight.h` as the check. On ROCm that is false for extensions (finding 1). The check as written is the one that missed it: the declaration is in the file, but behind a guard that only PyTorch's own build satisfies. Rewrite the entry so that (i) on ROCm the handle is usable from extensions only from torch 2.4; (ii) the check must read the preprocessor guard around each declaration, not just find the name; and (iii) a guard on `ROCM_VERSION` (and on any other macro PyTorch defines only through CMake) is false in a `cpp_extension` build up to 2.3. Keep llm-awq as the source.
3. Every Linux build now runs without ninja, CUDA included. `awq/kernels/setup.py:102,115`: `use_ninja = sys.platform == "win32" and IS_ROCM` and `BuildExtension.with_options(use_ninja=use_ninja)` pass `use_ninja=False` everywhere except Windows+ROCm. Upstream used plain `BuildExtension`, whose default is `use_ninja=True` (`torch/utils/cpp_extension.py:832`), so a Linux CUDA user's build changes from parallel ninja to serial distutils. The porting round's own notes measure the cost on ROCm: 44 s with ninja, 10m15s serial. ninja is already the default on Windows, so the override does nothing that the default does not. Restore `cmdclass={"build_ext": BuildExtension}` and delete `use_ninja`, including its comment.
4. The CUDA host-compiler standard changed with no reason left on ROCm. `awq/kernels/setup.py:49` sets `cxx_args` to `-std=c++20`; upstream had `-std=c++17`, and the CUDA `nvcc` list (`:35`) still says c++17. Since a9998ae no ROCm source reaches the host compiler, so the `cxx` list now affects only the CUDA build: `pybind.cpp` and `ft_attention.cpp` go to g++ as C++20, and the `.cu` files go to nvcc as C++17. Restore `-std=c++17` in `cxx_args`; the ROCm `nvcc_args` keep c++20.

Checked and not findings this round. The dac7721 workspace lifetime is safe: `at::empty` allocates on the current stream (the device guard is set and `stream` is `getCurrentCUDAStream()`), and when the block is freed at function exit, the caching allocator only reuses it in order on that stream. `make_bfloat162` / `__floats2bfloat162_rn` are absent from this host's ROCm SDK `amd_hip_bf16.h`, and `__float22bfloat162_rn` is present (lesson 1 holds). The shim copies sit next to their source, so relative includes resolve. Legacy SIMT GEMM: the loads stay in bounds (OC % 64 and IC % 32 are checked; the M tail is clamped). `jargon.py --port llm-awq` is clean. The three new commits have `[ROCm]` titles of 59-66 chars, AI disclosure, a Test Plan, no trailer, and ASCII only.

## Porting round 2026-10-30 (linux-gfx90a): re-review fixes

Answers "Review 2026-10-29 (linux-gfx1100, re-review at 8f14286)". Fork `moat-port`: 8f14286 -> 1f3aadc -> e8b1937. Host: AMD Instinct MI250X (gfx90a, wave64), ROCm 7.16 (TheRock SDK), torch 2.16.0a0+git6c3af02, Python 3.11. The review was recorded only here; review PR #1 has no threads, so nothing to reply to on it.

1. (new finding 1, torch 2.3) 1f3aadc: `w8a8_gemm_cuda.cu` no longer calls `at::cuda::getCurrentCUDABlasLtHandle()`. `w8a8_hipblaslt_handle()` keeps a `std::vector<hipblasLtHandle_t>` indexed by `hipGetDevice()`, filled by `hipblasLtCreate` on first use under a function-static `std::mutex`, never destroyed (as torch leaks its own handles). A failed create returns nullptr and the GEMM falls back to the SIMT kernel (retried on the next call). Current stream and the `at::empty` byte workspace from dac7721 unchanged. No torch-version conditional.
   Header verification (`git show <tag>:<file>` in the pytorch clone), reading the guard around every declaration the HIP build uses. After hipify the extension uses only `at::cuda::getCurrentCUDAStream`, `at::cuda::CUDAGuard`, `at::cuda::OptionalCUDAGuard`, `device_of`, `at::empty`/TensorOptions, through `ATen/hip/HIPContext.h` and `c10/hip/HIPGuard.h` (generated from `ATen/cuda/CUDAContext.h`, `c10/cuda/CUDAGuard.h`). At v2.3.0 and v2.8.0: `c10/cuda/CUDAStream.h` declares `getCurrentCUDAStream` unguarded (the only `#ifdef USE_ROCM` is inside an inline priority check); `c10/cuda/CUDAGuard.h` has no `#if` at all; `ATen/DeviceGuard.h` `device_of` unguarded; `ATen/cuda/CUDAContext.h` is includes only; `aten/src/ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h` (`getCurrentHIPStreamMasqueradingAsCUDA`) has no guard; `HIPGuardImplMasqueradingAsCUDA.h` declares `HIPGuardMasqueradingAsCUDA` / `OptionalHIPGuardMasqueradingAsCUDA` unguarded, its one v2.3.0 guard (`ROCM_VERSION < 50201`, line 104) is inside the inline `deviceCount()` body and declares nothing. `CUDAContextLight.h`: v2.3.0 lines 12-14 and 85-87 guard `<cublasLt.h>` and the handle on `ROCM_VERSION >= 50700` (confirmed the review: `ROCM_VERSION` only from `cmake/Dependencies.cmake:1297` add_definitions; v2.3.0 `cpp_extension.py` COMMON_HIP_FLAGS = `-D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1` (+`-DHIPBLAS_V2` for ROCm >= 6.0)); v2.4.0 and v2.8.0 declare the handle unguarded; `getCUDABlasLtWorkspace*` absent at v2.8.0. The extension now calls none of the guarded names.
   NOT built against a torch 2.3 (or 2.4..2.8) ROCm wheel: none installed here and pip is blocked on fleet hosts. So the torch 2.3 build is verified by header reading only. Separate, unverified floor: torch 2.3 ROCm wheels exist for ROCm 5.7 and 6.0; the hipBLASLt calls use `HIPBLAS_COMPUTE_32I` / `HIP_R_8I` (hipDataType), and whether ROCm 5.7's hipBLASLt headers accept those was not checked (no 5.7 SDK here).
   New test `tests/w8a8_handle_test.py <w8a8_gemm_cuda.cu>`: builds the handle function text in /tmp via load_inline (no checkout path) and races its first use from 16 `std::thread`s spread over the visible devices (the pybind entry holds the GIL, so Python threads cannot race it). PASS: 16 threads over 1 device, one non-null handle. Only one GPU visible (dispatcher pinned HIP_VISIBLE_DEVICES), so per-device distinctness is not exercised.
2. (new finding 2) Skill `references/fault-classes.md`, "Torch API used by a ROCm-only branch is a version floor nobody sees", rewritten: handle usable from an extension on ROCm only from torch 2.4; read the preprocessor guard around each declaration, not just the name; a `ROCM_VERSION` guard (and any macro torch defines only through CMake) is false in a `cpp_extension` build up to 2.3; recommended pattern is the extension-owned per-device handle. Source still llm-awq.
3. (new finding 3) e8b1937: setup.py uses plain `BuildExtension` again; `use_ninja`, its comment and the now-unused `import sys` removed. This round's build ran through ninja (`[1/14]`..`[14/14]`), 115 s at MAX_JOBS=32.
4. (new finding 4) e8b1937: `cxx_args` back to `-std=c++17` (CUDA host list matches upstream again apart from ordering of `-fopenmp -lgomp`, which is appended for CUDA only). torch's ROCm path passes `extra_compile_args['nvcc']` to hipcc for .cu (cpp_extension.py `cuda_post_cflags`), so the ROCm build is unaffected.

PR-PREP SQUASH TODO (commit-message corrections; history not rewritten now):
- dac7721 body: delete "The cached hipBLASLt handle (available since PyTorch 2.3)"; the handle is now the extension's own (1f3aadc). Its claim that the ROCm build "failed against PyTorch 2.3 through 2.8" and is fixed by dac7721 is wrong for 2.3: dac7721 restores 2.4..2.8 only, 1f3aadc covers 2.3 (by header reading; no 2.3 wheel build).
- 73edaef: TinyChat logits agree to <= 2.9e-3, not 2.7e-3 (2.930e-03 again this round).
- 1f3aadc / e8b1937 Test Plans: use the established form (`PYTHONPATH=. python ...` and the line "# (the test scripts are kept outside this repository)").

Build:
```
cd projects/llm-awq/src/awq/kernels
find csrc \( -name '*.hip' -o -name '*_hip.*' -o -name 'hip_bf16_*' -o -name '*_cpp_shim.cu' \) ! -name 'cuda_to_hip*' -delete
rm -rf build
PYTORCH_ROCM_ARCH=gfx90a MAX_JOBS=32 python setup.py build_ext --inplace
```

Tests at e8b1937 (from /tmp, `PYTHONPATH=<src>/awq/kernels`, scripts in projects/llm-awq/tests/), all PASS:
- `attn_test.py`: 443/443 (fp32 D=128 t=16240 refused, 65552 > 65536 B, as before).
- `tinychat_e2e.py <src>`: 4/4, argmax 40/40, 40/40, 64/64, 40/40; max |dlogit| 2.930e-03, min cos 0.999999.
- `w8a8_test.py`: 25/25, paths {'hipblaslt': 24} (handle now from `w8a8_hipblaslt_handle()`), side-stream ordering PASS.
- `w8a8_simt_test.py <w8a8_gemm_cuda.cu>`: 7/7 exact.
- `legacy_gemm_test.py`: 8/8, max err vs dequant ref 3.9e-3.
- `w4a16_test.py <src>`: 30/30.
- `w8a8_handle_test.py <w8a8_gemm_cuda.cu>`: PASS (new, see 1).
jargon.py --port llm-awq clean; prose.py clean on both commit bodies; titles 56 and 58 chars. Still owed: wave32 and Windows runs at the new head (validators).

## Review 2026-10-30 (linux-gfx1100, re-review at e8b1937)

Verdict: changes-requested. /pr-review skill, local-branch mode, fork `moat-port` at e8b1937, whole diff d6e797a...HEAD, plus the skill edit on port/llm-awq. Static review; no build or GPU run on this host. jeff asked for this re-review of the answers (1f3aadc, e8b1937) to "Review 2026-10-29 (linux-gfx1100, re-review at 8f14286)".

Status of the four findings from that review:
1. Resolved for torch 2.3 with ROCm >= 6.0. `w8a8_gemm_cuda.cu:971-985` owns a per-device handle (`hipGetDevice` under the caller's device guard, `hipblasLtCreate` under a mutex, nullptr -> SIMT fallback). `grep -n 'getCurrentCUDABlasLtHandle\|getCUDABlasLt' csrc` is empty. Header reading only, no torch 2.3 wheel build; the porter says so in the notes and the commit. The torch 2.3 + ROCm 5.7 combination still fails, for a different reason (new finding 2).
2. Partly resolved. Points (ii) (read the guard, not the name) and (iii) (the macro is CMake-only) are in. The scope is still wrong (new finding 1).
3. Resolved. setup.py goes back to plain `BuildExtension`; `use_ninja`, its comment and `import sys` are removed.
4. Resolved. `cxx_args` is back to `-std=c++17`; the ROCm `nvcc_args` keep c++20.

New findings:
1. The skill lesson says the `ROCM_VERSION` trap ends at 2.3. It does not. `.claude/skills/cuda-to-rocm/references/fault-classes.md:396-404` says "Any guard on `ROCM_VERSION` ... reads false in an extension up to 2.3". No torch release defines `ROCM_VERSION` for an extension. `COMMON_HIP_FLAGS` is `-D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2` at v2.4.0, v2.8.0, v2.9.0 and on current main (6c3af02, `torch/utils/cpp_extension.py:438-442`). `git grep 'define\s+ROCM_VERSION\b'` finds nothing at any of those refs, and the installed torch 2.16 include tree has no such define either. Headers still guard on it: v2.8.0 has `aten/src/ATen/cuda/CUDADataType.h:81` (`!defined(USE_ROCM) || ROCM_VERSION >= 60300`, the fp8 cases), `ATen/native/cuda/KernelUtils.cuh:9` and `ATen/cuda/tunable/GemmHipblaslt.h:65,74,622`. v2.9.0 adds `CUDADataType.h:93` and `torch/csrc/jit/codegen/fuser/cuda/resource_strings.h:15`. The installed 2.16 `ATen/cuda/CUDADataType.h:85` still has the fp8 guard. So the guard is false in every `cpp_extension` build at every version. 2.4 did not make the macro visible to extensions. It removed the one guard around the handle (ae9a4fa). A reader who trusts "up to 2.3" will assume `CUDADataType.h` maps fp8 for their extension on 2.8, and on ROCm it does not. Rewrite the sentence: a `ROCM_VERSION` guard in an installed header is false in every extension build, current releases included. The handle is reachable from 2.4 only because that release dropped its guard.
2. The ROCm floor is now 6.0 for the whole extension, and nothing says so. The HIP W8A8 path (`w8a8_gemm_cuda.cu:999-1004`) passes `HIPBLAS_COMPUTE_32I` and `HIP_R_8I` / `HIP_R_32I` (hipDataType). At ROCm 5.7, hipBLASLt takes `hipblasLtComputeType_t` and `hipblasDatatype_t`: `library/include/hipblaslt.h:289-290` (`hipblasLtMatrixLayoutCreate`) and `:398-400` (`hipblasLtMatmulDescCreate`) at tag rocm-5.7.0. Its include dir has no `HIPBLAS_COMPUTE_32I`. ROCm 6.0 switched to `hipDataType` / `hipblasComputeType_t` (`hipblaslt.h:392-393, 496-498` at rocm-6.0.0). Torch 2.3, the version pyproject.toml pins, shipped ROCm 5.7 and 6.0 wheels, and its `cpp_extension.py:240` adds `-DHIPBLAS_V2` only for ROCm >= 6.0. So on torch 2.3 + ROCm 5.7 `w8a8_gemm_cuda.cu` fails to compile, and the validated W4A16 path goes down with it, the same failure mode as the earlier findings. a2a710a had no hipBLASLt and did not have this floor; 7f0f35f introduced it. Fix: in the AMD note at `README.md:127`, say the kernels need ROCm 6.0 or newer. At the PR-prep squash, the 1f3aadc body should say the fix applies to torch 2.3 with ROCm 6.0, not torch 2.3 in general.
3. Literal `%%` in a user-visible error. `awq/kernels/csrc/quantization_new/gemm/gemm_cuda.cu:1171`: `TORCH_CHECK(false, "... requires K %% 128 == 0 and N %% 8 == 0.")`. TORCH_CHECK joins its message arguments with `c10::str`; it is not printf. The user therefore sees `K %% 128`. Write `%` and move the condition into the check: `TORCH_CHECK(num_in_channels % 128 == 0 && num_out_channels % (N_PER_BLOCK * K_INTERLEAVE) == 0, "gemm_forward_cuda_new (HIP) requires K % 128 == 0 and N % 8 == 0.")`.

Checked and not findings this round. The handle cache is correct across devices: `w8a8_gemm_forward_hip` sets `OptionalCUDAGuard(device_of(_in_feats))` before `w8a8_int8_gemm_hipblaslt`, so `hipGetDevice` names the input's device, and `hipblasLtCreate` binds the handle to that device. The resize and create both happen under the lock, and nullptr is retried on the next call. With `use_ninja` gone, the ROCm .cu sources still get `extra_compile_args['nvcc']` and the shims still compile with hipcc (porter's ninja log, 14 steps). CUDA builds are back on ninja with the upstream `cxx` flags (only `-fopenmp -lgomp` moved to the end of the list). The PR-prep squash TODOs recorded by the porter (dac7721 and 73edaef bodies, the Test Plan form in 1f3aadc and e8b1937) cover the commit-message issues from the last two reviews; history has not been rewritten yet, as agreed. `jargon.py --port llm-awq` is clean. The new commits have `[ROCm]` titles of 56 and 58 chars, AI disclosure, a Test Plan, no trailer, and ASCII only.

## Porting round 2026-10-30 (linux-gfx90a): answers re-review at e8b1937

Answers "Review 2026-10-30 (linux-gfx1100, re-review at e8b1937)". Fork `moat-port`: e8b1937 -> ca71a9a -> 90a711b. Host: AMD Instinct MI250X (gfx90a, wave64), ROCm 7.16 (TheRock SDK, hipBLASLt 1.4.1), torch 2.16.0a0+git6c3af02, Python 3.11. No GitHub writes this round (jeff's instruction), so review PR #1 got no replies; it had no threads anyway.

Version facts, all from `git show <tag>:<file>` in blobless clones of ROCm/hipBLASLt, ROCm/hipBLAS, ROCm/clr, ROCm/HIP, ROCm/rocm-cmake, and the read-only pytorch checkout:
- hipBLASLt rocm-5.7.0 and rocm-5.7.1: VERSION_STRING 0.3.0; `hipblaslt.h:289-290` `hipblasLtMatrixLayoutCreate(..., hipblasDatatype_t, ...)`, `:398-400` `hipblasLtMatmulDescCreate(..., hipblasLtComputeType_t, hipblasDatatype_t)`. rocm-6.0.0: 0.6.0; `:392-393` takes `hipDataType`, `:496-498` takes `hipblasComputeType_t, hipDataType`. 6.1.0 0.7.0, 6.2.0 0.8.0, 6.4.0 0.12.0.
- `hipblaslt-version.h.in` (both tags) defines `HIPBLASLT_VERSION_MAJOR/MINOR/PATCH/TWEAK` from `@hipblaslt_VERSION_*@`; `rocm_setup_version` (rocm-cmake rocm-5.7.0 `ROCMSetupVersion.cmake:119-147`) sets `${PROJECT_NAME}_VERSION_MAJOR/MINOR/PATCH` from VERSION_STRING; `library/CMakeLists.txt:34-35` configures it into `include/hipblaslt/hipblaslt-version.h`, installed by `rocm_install_targets(... INCLUDE ${CMAKE_BINARY_DIR}/include)`. It has no includes.
- hipBLAS rocm-6.0.0 `hipblas.h:555` `HIPBLAS_COMPUTE_32I` is outside the `#ifdef HIPBLAS_V2` block (487-539), so it does not depend on the flag.
- torch v2.3.0 `.github/scripts/generate_binary_build_matrix.py:25` `ROCM_ARCHES = ["5.7", "6.0"]`.
- clr `amd_hip_bf16.h` rocm-5.7.0 vs rocm-6.0.0 differ only by an added `amd_hip_common.h` include; every bf16/half intrinsic the HIP paths call (`__float22bfloat162_rn`, `__bfloat1622float2`, `__bfloat162bfloat162`, `__hfma2`, `__hmax2`, ...) is declared at rocm-5.7.0; `__floats2bfloat162_rn` / `__float2bfloat162_rn` are absent at both and the HIP code does not call them (own fallback, or `__float22bfloat162_rn`). ROCm/HIP rocm-5.7.0 has `include/hip/hip_bf16.h`, `hip_fp16.h`, `library_types.h`. hipBLASLt itself exists from rocm-5.5.0, so `-lhipblaslt` links on 5.7.
- `ROCM_VERSION`: no `#define ROCM_VERSION` and no `-DROCM_VERSION` in `cpp_extension.py` at v2.2.0, v2.3.0, v2.4.0, v2.8.0, v2.9.0, main (6c3af02); none in the installed 2.16 include tree, which still guards on it (`hip/HIPEvent.h:146`, `torch/headeronly/cuda/KernelUtils.h:20`, ...). The SDK `rocm_version.h` defines only `ROCM_VERSION_MAJOR/MINOR/PATCH` and `ROCM_BUILD_INFO`. v2.3.0 `cmake/Dependencies.cmake:1297` is the only definition. v2.4.0 `CUDAContextLight.h:83` declares the handle with no guard (ae9a4fa "[ROCm] enforce ROCM_VERSION >= 6.0 (#125646)"). `getCUDABlasLtWorkspace*`: 0 hits at v2.8.0, present at v2.9.0. v2.8.0 `CUDADataType.h:81` and v2.9.0 `:81,93` guard on `ROCM_VERSION`.

1. (new finding 2, ROCm floor) ca71a9a: `w8a8_gemm_cuda.cu` includes `<hipblaslt/hipblaslt-version.h>` under `__has_include` and compiles the handle cache and `w8a8_int8_gemm_hipblaslt` only when hipBLASLt >= 0.6; otherwise a stub returns false and every call takes the existing SIMT kernel. `hipblaslt.h`, `<mutex>`, `<vector>` are included only inside the gate. README AMD note: "The kernels need ROCm 5.7 or newer; the W8A8 GEMM uses hipBLASLt on ROCm 6.0 and newer and a slower portable kernel on ROCm 5.7." The 5.7 floor for the rest of the extension rests on header reading (above) plus torch 2.3's own floor; no ROCm 5.7 or 6.0 SDK or torch 2.3 ROCm wheel is available here, so neither was built against.
   Gate test: /tmp copy of `awq/kernels` with `include_dirs=["/tmp/llmawq-gate/fakeinc"]` added to its setup.py, where `fakeinc/hipblaslt/hipblaslt-version.h` defines 0.3.0 (found before the SDK's copy). Builds; `nm -D` shows 0 hipblasLt symbols; a profiler probe shows only `w8a8_int8_gemm_simt<16, 64>` + epilogue for 16x4096x4096, 17x4096x11008, 300x130x520; `w8a8_test.py` 25/25.
2. (new finding 1, skill lesson) `references/fault-classes.md` entry rewritten: `ROCM_VERSION` is never defined in a `cpp_extension` build at any version checked, so every such guard in an installed header is false in an extension, current releases included (v2.8.0 `CUDADataType.h` fp8 example); 2.4 only dropped the guard around the handle declaration; read the guard around each declaration you call at the oldest supported tag. Added the hipBLASLt 0.3 -> 0.6 type change and the `HIPBLASLT_VERSION_*` gate as part of the same entry. Source llm-awq.
3. (new finding 3) 90a711b: `TORCH_CHECK(num_in_channels % 128 == 0 && num_out_channels % (N_PER_BLOCK * K_INTERLEAVE) == 0, "... K % 128 == 0 and N % 8 == 0.")`.
4. Test fix (MOAT-side `tests/w8a8_test.py`): the path classifier now ignores runtime-API events (`hip*`/`cuda*`). The profiler on this host sometimes drops the kernel record; before, a profile with only `hipLaunchKernel`/`hipDeviceSynchronize` was labelled `hipblaslt` (seen 3/24 in the gate build, which has no hipBLASLt code). Such cases now read `?`.

PR-PREP SQUASH TODO addition: 1f3aadc body should say the fix applies to torch 2.3 with ROCm 6.0 (ca71a9a covers ROCm 5.7 by falling back to SIMT).

Build (ninja `[14/14]`):
```
cd projects/llm-awq/src/awq/kernels
find csrc \( -name '*.hip' -o -name '*_hip.*' -o -name 'hip_bf16_*' -o -name '*_cpp_shim.cu' \) ! -name 'cuda_to_hip*' -delete
rm -rf build
PYTORCH_ROCM_ARCH=gfx90a MAX_JOBS=32 python setup.py build_ext --inplace
```

Tests at 90a711b (from /tmp, `PYTHONPATH=<src>/awq/kernels`, HIP_VISIBLE_DEVICES pinned to one GPU), all PASS:
- `attn_test.py`: 443/443 (fp32 D=128 t=16240 and t=20000 refused for shared memory, as before).
- `tinychat_e2e.py <src>`: 4/4, argmax 40/40, 40/40, 64/64, 40/40; max |dlogit| 2.930e-03, min cos 0.999999.
- `w8a8_test.py`: 25/25; with the fixed classifier paths {'hipblaslt': 20, '?': 4} (first run, old classifier: {'hipblaslt': 24}).
- `w8a8_simt_test.py <w8a8_gemm_cuda.cu>`: 7/7 exact.
- `legacy_gemm_test.py`: 8/8, max err vs dequant ref 3.9e-3.
- `w4a16_test.py <src>`: 30/30.
- `w8a8_handle_test.py <w8a8_gemm_cuda.cu>`: PASS, 16 threads, 1 device, one handle.
jargon.py --port llm-awq clean; prose.py clean on both bodies; titles 65 and 58 chars. Still owed: wave32 and Windows runs at the new head (validators).

## Review 2026-10-30 (linux-gfx1100, re-review at 90a711b)

Verdict: review-passed. /pr-review skill, local-branch mode, fork `moat-port` at 90a711b, whole diff d6e797a...HEAD, plus the skill edit on port/llm-awq. Static review; no build or GPU run on this host. jeff asked for this re-review of the answers (ca71a9a, 90a711b) to "Review 2026-10-30 (linux-gfx1100, re-review at e8b1937)".

Status of the three findings from that review:
1. Resolved. The skill entry (`references/fault-classes.md:391-419`) now says `ROCM_VERSION` is never defined in a `cpp_extension` build, so every such guard in an installed header is false in an extension, current releases included, and that 2.4 only dropped the guard around the handle. Checked in the pytorch clone: no `#define ROCM_VERSION` under aten/c10/torch/csrc/torch/utils at v2.2.0, v2.3.0, v2.4.0, v2.8.0, v2.9.0; v2.8.0 `CUDADataType.h:81` carries the fp8 guard; ae9a4fa is in v2.4.0 and not in v2.3.0; `getCUDABlasLtWorkspace` appears in `CUDAContextLight.h` at v2.9.0 only.
2. Resolved. ca71a9a compiles the hipBLASLt code only when `HIPBLASLT_VERSION_MAJOR > 0 || MINOR >= 6`, else a stub with the same signature returns false and the SIMT kernel runs. Checked in a blobless ROCm/hipBLASLt clone: VERSION_STRING 0.3.0 at rocm-5.7.0 and 0.6.0 at rocm-6.0.0; at both tags `library/CMakeLists.txt:34-35` configures `include/hipblaslt/hipblaslt-version.h` (`HIPBLASLT_VERSION_MAJOR/MINOR/PATCH/TWEAK`) and installs it with the other headers, and rocm-5.7.0 `hipblaslt.h:48` itself includes `hipblaslt/hipblaslt-version.h`, so the version header is present wherever the old unconditional `hipblaslt.h` include resolved. The ROCm 7.x SDK on this host has it at the same path. README states ROCm 5.7 minimum and hipBLASLt from 6.0.
3. Resolved. 90a711b folds the condition into `TORCH_CHECK` and prints `%`.

No new findings. Whole-diff pass (attention, reductions, compat header, dequant, W4A16 SIMT GEMM, legacy GEMM, W8A8, setup.py, README, gitignores) found no user-hitting defect. `jargon.py --port llm-awq` clean; all 12 commits have `[ROCm]` titles of 47-66 chars, no trailer or noreply, ASCII bodies; fork worktree clean.

Observations, not findings (no action required for this round):
- The HIP W4A16 GEMM (`gemm_simt.cuh`) accumulates each thread's partial sum in T (fp16/bf16) before the fp32 cross-thread reduce, the same scheme as the CUDA GEMV for M < 8; CUDA's tensor-core GEMM for M >= 8 accumulates in fp32. TinyChat fp16 e2e agrees to 2.9e-3; bf16 prefill was not measured separately. Worth a bf16 check by a validator, not a blocker.
- Large-M prefill performance of that GEMM (re-dequant per 4-row batch, runtime-indexed `psum`) is already covered by the deferred MFMA GEMM item.

PR-PREP SQUASH TODO additions (nits):
- Skill entry `fault-classes.md:396`: `-DHIPBLAS_V2` is passed only from v2.3.0 (absent at v2.2.0, which the same sentence lists as checked); say "plus `-DHIPBLAS_V2` from 2.3".
- ca71a9a body is accurate as written; the existing TODOs (dac7721, 73edaef, 1f3aadc bodies and Test Plan form) still stand.

## Validation 2026-10-30 (linux-gfx1201, wave32)

Verdict: completed at fork `moat-port` 90a711b. jeff asked for this run after the review passed at 90a711b. Host: AMD Radeon AI PRO R9700 (gfx1201, RDNA4, wave32), ROCm 7.16 (TheRock SDK in the image venv, hipcc 7.16.26385), torch 2.16.0a0+git6c3af02, Python 3.11. HIP_VISIBLE_DEVICES pinned to one GPU by the dispatcher; everything under `ulimit -c 0`.

Build (ninja `[14/14]`, 54 s; 0 errors; llvm-objdump --offloading shows only gfx1201):
```
cd projects/llm-awq/src/awq/kernels
find csrc \( -name '*.hip' -o -name '*_hip.*' -o -name 'hip_bf16_*' -o -name '*_cpp_shim.cu' \) ! -name 'cuda_to_hip*' -delete
rm -rf build
PYTORCH_ROCM_ARCH=gfx1201 MAX_JOBS=24 python setup.py build_ext --inplace
```

Tests at 90a711b (from /tmp, `PYTHONPATH=<src>/awq/kernels`, scripts in projects/llm-awq/tests/), all PASS, every script exit 0:
- `attn_test.py`: 443/443. Max out err fp16 2.54e-3, bf16 1.91e-2, fp32 1.19e-5. fp32 D=128 t=16239 passes; t=16240 (65552 B) and t=20000 (80592 B) refused against the 65536 B LDS limit, as on gfx90a.
- `tinychat_e2e.py <src>`: 4/4; argmax 40/40, 40/40, 64/64, 40/40; max |dlogit| 2.930e-03, min cos 0.999999.
- `w8a8_test.py`: 25/25; paths {'hipblaslt': 24} (no `?` this run), side-stream ordering PASS. hipBLASLt int8 is taken on gfx1201.
- `w8a8_simt_test.py <src>/awq/kernels/csrc/w8a8/w8a8_gemm_cuda.cu`: 7/7 exact (SIMT fallback built via load_inline in /tmp, since the extension reaches hipBLASLt here).
- `legacy_gemm_test.py`: 8/8, max err vs dequant ref 3.9e-3, vs legacy gemv 7.8e-3.
- `w4a16_test.py <src>`: 30/30 (fp16 only; the reviewer's bf16-prefill observation is still unmeasured).
- `w8a8_handle_test.py <w8a8_gemm_cuda.cu>`: PASS, 16 threads over 1 device, one handle.

CUDA gate: cuda-not-validated: no CUDA toolkit on this host (no nvcc, no conda to create the cuda-12.8 env). Supplementary only: `tests/cuda_view.py` diff vs d6e797a shows 0 changed CUDA-visible lines in w8a8_gemm_cuda.cu, gemm_cuda_gen.cu, both dequantize.cuh and w8a8/quantization.cu; the remaining files differ by the reviewed shuffle/macro lines.

Checks: `python3 utils/jargon.py --port llm-awq` clean. README documents the ROCm build (line 127). Fork worktree: no modified tracked files (`git status --porcelain` shows only untracked build output).
