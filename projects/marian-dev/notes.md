# marian-dev notes

## Port summary (linux-gfx90a, lead)
Strategy A: a single `src/tensors/gpu/cuda_to_hip.h` compat header (runtime
cuda*->hip* aliases + a CUDA_VERSION>=11000 sentinel), plus forwarding shim
headers under `src/tensors/gpu/hip_compat/` (`cuda.h`, `cuda_runtime.h`,
`cuda_fp16.h`, `cublas_v2.h`, `cublasLt.h`, `cusparse.h`, `cusparse_v2.h`,
`curand.h`, `cooperative_groups.h`). The legacy FindCUDA `cuda_add_library`
call is satisfied by a cupoch-style shim macro under `option(USE_HIP)` that
marks the .cu and mixed .cpp TUs `LANGUAGE HIP`. NVIDIA path is unchanged
(every divergence is `#if defined(USE_HIP)` / `else`).

Fork: https://github.com/AMD-Ecosystem/marian-dev (branch moat-port). Actions
disabled on the fork.

## Build (gfx90a)
ROCm 7.2.1 at /opt/rocm. CMake 4.x needs the policy-min shim. Init only the
sentencepiece submodule (NOT nccl). Build in a scratch dir, never the repo root.
```
cd projects/marian-dev/src
git submodule update --init src/3rd_party/sentencepiece
export HIP_VISIBLE_DEVICES=3 ROCM_PATH=/opt/rocm
cmake -S . -B build-hip -G Ninja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCOMPILE_CUDA=ON -DUSE_CUDNN=OFF -DUSE_NCCL=OFF \
  -DUSE_FBGEMM=OFF -DCOMPILE_CPU=ON -DCMAKE_BUILD_TYPE=Release \
  -DCOMPILE_TESTS=ON -DUSE_MKL=OFF -DUSE_TCMALLOC=OFF -DUSE_DOXYGEN=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DBUILD_ARCH=native
cmake --build build-hip -j
```
Followers (gfx1100/gfx1151): change only `-DCMAKE_HIP_ARCHITECTURES`; no source
edit (HIP_ARCHITECTURES drives --offload-arch; the wave64 fix is wave-agnostic).

## THE correctness fix (wave64): topk.cu + nth_element.cu
Both used the classic warp-synchronous unrolled reduction tail
(`UNROLL_MAXARG_LOOP(32/16/.../1)`) with no barrier on non-volatile shared
memory below 32 lanes. On CDNA wave64 the low 32 lanes of a 64-lane wavefront
are NOT lockstep, so that tail races -> non-deterministic argmax. nth_element is
the beam-search top-k on the decode path, so a wrong result silently corrupts
translations. Fixed (USE_HIP-guarded): drop the unrolled tail and run the
`__syncthreads()`-synchronized tree to `s>0` (same compare order, block-wide
barrier, correct on any wave size). CUDA path byte-identical.

Verified on gfx90a (GCD 3): operator_tests topk section (exact top-k values +
argmax/argmin/sort) passes; end-to-end beam-search decode is bit-identical
run-to-run AND matches the CPU decode exactly; `gMaxElement`/`gMaxElementUpdate`
confirmed dispatched on gfx90a via AMD_LOG_LEVEL=3.

## Validation (real gfx90a, GCD 3)
- Unit suites PASS: graph_tests, attention_tests, transformer_tests,
  operator_tests (284/287 assertions; the 3 failures are the cuSPARSE csr-dot
  path only -- see follow-ups). These exercise element-wise ops, softmax,
  layernorm, the reduce_all.h block reduction (wave64-correct as-is),
  prod/affine (hipBLAS + the hipBLASLt bias/ReLU epilogue), attention, and the
  full Transformer fwd+bwd.
- End-to-end: trained a tiny Transformer (reverse-copy toy task) to convergence
  on GPU, then beam-search decoded. GPU run1==run2 (deterministic) and GPU==CPU
  output. Commands:
```
# train (word vocab to avoid spm vocab-size floor on a toy corpus)
marian --type transformer -t train.src train.tgt -m model.npz \
  --vocabs vocab.src.yml vocab.tgt.yml --dim-emb 64 --transformer-dim-ffn 128 \
  --transformer-heads 2 --enc-depth 2 --dec-depth 2 --after 600u --devices 0
# decode twice on GPU + once on CPU, then diff
marian-decoder -m model.npz -v vocab.src.yml vocab.tgt.yml -i test.src -b 6 --devices 0
marian-decoder -m model.npz -v vocab.src.yml vocab.tgt.yml -i test.src -b 6 --cpu-threads 1
```
- rnn_tests: the 3 hardcoded-reference comparisons differ because hipRAND and
  cuRAND produce different streams for the same seed (the references at
  rnn_tests.cpp are cuRAND-specific glorotUniform draws). The kernels are
  correct; the 21 shape/structure assertions pass. Not a port bug.

## Library mapping
- cuBLAS->hipBLAS, cuBLASLt->hipBLASLt, cuSPARSE->hipSPARSE, cuRAND->hipRAND,
  Thrust/CUB->rocThrust (thrust::cuda::par -> thrust::hip::par under USE_HIP).
- hipBLASLt deltas handled in prod.cpp (all USE_HIP-guarded):
  - The `MATMUL_PREF_MIN_ALIGNMENT_{A,B,C,D}_BYTES` preference attrs do NOT
    exist in hipBLASLt -> dropped.
  - `hipblasLtMatmulDescCreate` rejects HIPBLAS_COMPUTE_16F and a 16F scale
    type: use HIPBLAS_COMPUTE_32F as the compute type and HIP_R_32F as the desc
    scale type for both fp16 and fp32 (matrix dtypes still set on the layouts).
  - `cublasSetMathMode(CUBLAS_TENSOR_OP_MATH)` returns "not supported" on
    gfx90a; setTensorMode/tensorOpsEnabled/unsetTensorMode are no-ops on HIP
    (gfx90a drives its matrix cores from the GEMM data/compute types directly).
  - cublasGemmEx compute-type slot -> HIPBLAS_COMPUTE_* (vs cudaDataType on
    CUDA); hipblasGemmBatchedEx array args are `const void*[]` (vs
    `const void* const[]`) -> per-platform cast macros.
  - cublasLt handle: CUDA encapsulates an lt handle inside the cublas handle;
    hipBLASLt needs a dedicated handle -> Backend::getCublasLtHandle() on HIP.

## __CUDACC__ / FP16 steering (important gotcha)
Do NOT globally `#define __CUDACC__` to take Marian's FP16 path: rocThrust keys
its device-system detection on `__CUDACC__` and would select its (unimplemented)
CUDA/CUB backend, producing "unimplemented for this system" template errors.
Instead the compat header only defines CUDA_VERSION, and types.h / operators.h /
defs.h / tensor.h / cpu/element.h were edited to treat `__HIPCC__` like
`__CUDACC__` for the host-intrinsics + FP16 steering. fp16 device-vs-host data
selection uses `__HIP_DEVICE_COMPILE__` (e.g. the LSH atomicAdd_block guards in
tensor_operators.cu; `atomicAdd_block` maps to `atomicAdd` on HIP).

Other gotchas:
- `halfx2(1.f)` is ambiguous on HIP (both __half and __half2 take a scalar) ->
  added a USE_HIP-only `halfx2(float)` ctor.
- alibi.cu used `thrust::tie`/`thrust::tuple` in device code (not device-callable
  on rocThrust) -> a trivial POD pair under USE_HIP.
- topk.cu's CUB segmented-sort path is not ported; Sort() falls back to the
  rocThrust sort_by_key path on HIP (only used by tests).
- The compat header is force-included on every HIP TU via CMAKE_HIP_FLAGS and on
  Marian's own host targets (lib/exe/tests) via target flags -- NOT globally,
  because the gnu++11 3rd_party libs (intgemm) choke on the HIP headers.
- HIPBLASLT_USE_ROCROLLER define is injected by find_package(hipblaslt); benign.

## Deferred follow-ups (SUPERSEDED 2026-08-10 -- historical, all three are done)
This section recorded the scope of the first round. The maintainer rejected that
scope (see "2026-08-10: Sent back by the maintainer") and all three features are
now implemented, built and exercised on GPU (see "2026-08-10: the three
scope-outs ported"): the hipSPARSE SpMM path is correct for fp32 and fp16, the
convolution/pooling path runs on MIOpen with USE_CUDNN=ON, the collectives run
on RCCL with USE_NCCL=ON, and the `pooling` app test is back in the HIP build.
Kept for history; read the 2026-08-10 sections for the current state.

1. cuSPARSE sparse-attention CSRProd (prod_sparse_cu11.h): hipSPARSE SpMM with
   the row/col-major + CSR_ALG2 setup gives wrong fp32 results and rejects fp16
   (HIPSPARSE_STATUS_NOT_SUPPORTED). This is the only failing unit path
   (operator_tests csr-dot). Used only by sparse-attention/LSH models, not the
   default dense Transformer. Needs hipSPARSE-specific order/alg work.
2. cuDNN convolution/pooling -> MIOpen: kept CUDA-only (USE_CUDNN=OFF). All
   cuDNN code is #ifdef CUDNN-gated; the char-CNN conv/pool path is not built.
   The `pooling` app test is dropped from the HIP build.
3. NCCL -> RCCL multi-GPU collectives: USE_NCCL=OFF for the lead (single-GPU
   train/decode validates the full kernel + BLAS surface). RCCL is API-
   compatible and ships in ROCm; a follow-up.

## Review 2026-06-02 (reviewer, linux-gfx90a, fork moat-port 25f910c vs base c9f287d)
Verdict: review-passed. No changes requested. Strategy A (compat header + cuda_add_library LANGUAGE-HIP shim) is correct for this legacy-FindCUDA pure-CMake build; the NVIDIA and CPU paths are byte-identical (every divergence is USE_HIP / __HIPCC__ guarded, and __CUDACC__->__CUDACC__||__HIPCC__ extensions are no-ops when neither is defined).

Re-ran the GPU suites myself on gfx90a (GCD 3) after an incremental rebuild at HEAD: graph (10/10), attention (6/6), transformer (3/3) all pass; operator 284/287 with the only 3 failures isolated to the `csr-dot product` SECTION (operator_tests.cpp:539,609,610), i.e. the deferred cuSPARSE SpMM path -- the dense `dot product` (508) and `topk operations` (1026, exact-value top-k/argmax/argmin/sort) pass. End-to-end decode artifacts confirm the load-bearing wave64 fix: GPU run1==run2 (deterministic) AND GPU==CPU (correct beam-search top-k). The topk.cu/nth_element.cu fix replaces the warp-synchronous UNROLL_MAXARG tail with the __syncthreads tree run to s>0 -- a strict generalization (same compare order, same `tid+s<end` guard), correct on any wave size; CUDA path unchanged.

reduce_all.h is untouched and is wave64-correct as left: the `tid<64` fold is fenced by cg::sync before the `thread_rank()<32` block folds sdata[tid+32] and shfl_downs within a width-32 tiled_partition -- no reliance on sub-wavefront lockstep. No hardcoded 32/warpSize/shfl/ballot/lane-mask in any added kernel line. No textures/surfaces/managed memory (those fault classes N/A). alibi.cu POD-pair swap preserves field order. hipBLASLt deltas (dropped MIN_ALIGNMENT prefs, HIPBLAS_COMPUTE_32F + HIP_R_32F scale, SetMathMode no-op, dedicated Lt handle) and the GemmEx compute-type/batched-array cast macros are all USE_HIP-guarded and leave the CUDA call sites identical.

rnn_tests 3 failures (rnn_tests.cpp:49->93) are NOT a port bug: the test seeds inputs with inits::glorotUniform() and compares against a hardcoded `#ifdef CUDA_FOUND` reference; the HIP build defines CUDA_FOUND so it uses the cuRAND-stream reference, but hipRAND yields a different stream for the same seed. RNN forward math is correct (21 structural assertions pass). Worth an upstream follow-up note only.

Hygiene: title `[ROCm] Port Marian GPU backend to HIP for AMD GPUs` (50 chars), mentions Claude, no noreply trailer, no ghstack, no em-dash, Test Plan present; author the public account (user's own public email -- not an AMD-internal account); fork Actions disabled (enabled:false); fork/master == upstream c9f287d (clean mirror). Deferred cuSPARSE/cuDNN/NCCL items are documented, not silently broken.

Minor (non-blocking, no fix required): getCublasLtHandle() (backend.h:81) does not check hipblasLtCreate's return, matching the existing lazy-init cublasCreate pattern; cumsum.cu:62 has a 2-space-indented `#if` (cosmetic).

## Validation 2026-06-02 (linux-gfx90a, GCD 3, fork moat-port 25f910c)

Platform: AMD Instinct MI250X / MI250 (gfx90a), ROCm 7.2.1, HIP_VISIBLE_DEVICES=3.

### Build

Full clean build from committed source (292 targets):
```
cd projects/marian-dev/src
export HIP_VISIBLE_DEVICES=3 ROCM_PATH=/opt/rocm
cmake -S . -B build-hip -G Ninja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCOMPILE_CUDA=ON -DUSE_CUDNN=OFF -DUSE_NCCL=OFF \
  -DUSE_FBGEMM=OFF -DCOMPILE_CPU=ON -DCMAKE_BUILD_TYPE=Release \
  -DCOMPILE_TESTS=ON -DUSE_MKL=OFF -DUSE_TCMALLOC=OFF -DUSE_DOXYGEN=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DBUILD_ARCH=native
cmake --build build-hip -j$(nproc)
```
Result: 292/292 targets, no errors.

### Unit suites

```
BUILD=projects/marian-dev/src/build-hip
$BUILD/src/tests/units/run_graph_tests
$BUILD/src/tests/units/run_attention_tests
$BUILD/src/tests/units/run_transformer_tests
$BUILD/src/tests/units/run_operator_tests
```

- graph_tests: 10/10 assertions, 4 test cases -- PASS
- attention_tests: 6/6 assertions, 3 test cases -- PASS
- transformer_tests: 3/3 assertions, 3 test cases -- PASS
- operator_tests: 284/287 assertions; the 3 failures are all in the `csr-dot product` SECTION (operator_tests.cpp:539,609,610) -- the documented deferred cuSPARSE SpMM path. The dense `dot product` and `topk operations` sections (exact top-k values, argmax/argmin/sort) PASS.

### End-to-end gate (silent-corruption / determinism guard)

Trained a tiny Transformer (reverse-copy toy task) on GPU from scratch, then beam-search decoded (beam=6) twice on GPU and once on CPU:
```
E2E=agent_space/marian-validate
MARIAN=projects/marian-dev/src/build-hip/marian
DECODER=projects/marian-dev/src/build-hip/marian-decoder

$MARIAN --type transformer -t $E2E/train.src $E2E/train.tgt -m $E2E/model.npz \
  --vocabs $E2E/vocab.src.yml $E2E/vocab.tgt.yml --dim-emb 64 \
  --transformer-dim-ffn 128 --transformer-heads 2 --enc-depth 2 --dec-depth 2 \
  --after 600u --devices 0

$DECODER -m $E2E/model.npz -v $E2E/vocab.src.yml $E2E/vocab.tgt.yml \
  -i $E2E/test.src -b 6 --devices 0 > $E2E/gpu1.out 2>&1

$DECODER -m $E2E/model.npz -v $E2E/vocab.src.yml $E2E/vocab.tgt.yml \
  -i $E2E/test.src -b 6 --devices 0 > $E2E/gpu2.out 2>&1

$DECODER -m $E2E/model.npz -v $E2E/vocab.src.yml $E2E/vocab.tgt.yml \
  -i $E2E/test.src -b 6 --cpu-threads 1 > $E2E/cpu.out 2>&1
```
Result: GPU run1 == GPU run2 (bit-identical, deterministic) AND GPU == CPU (correct). diff exits 0 on both comparisons.

gMaxElement/gMaxElementUpdate kernel dispatch on gfx90a confirmed via AMD_LOG_LEVEL=3:
```
ShaderName : void marian::gMaxElement<float>(...)
ShaderName : void marian::gMaxElementUpdate<float>(...)
```

### Verdict: PASS -- linux-gfx90a completed at validated_sha 25f910c

## Validation 2026-06-02 (linux-gfx1100, AMD Radeon Pro W7800 48GB, wave32)

Platform: 2x AMD Radeon Pro W7800 48GB (gfx1100, RDNA3, wave32), ROCm 7.2.1,
HIP_VISIBLE_DEVICES=0. Fork moat-port @ 25f910c -- no source changes vs gfx90a
lead (validate-first follower, no delta-port needed).

### Build

Cloned AMD-Ecosystem/marian-dev @ moat-port (25f910c). Submodule: sentencepiece only
(not nccl). Build in scratch dir outside fork clone.

```
git submodule update --init src/3rd_party/sentencepiece
cmake -S /var/lib/jenkins/moat/projects/marian-dev/src \
  -B /var/lib/jenkins/moat/agent_space/marian-build-gfx1100 -G Ninja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCOMPILE_CUDA=ON -DUSE_CUDNN=OFF -DUSE_NCCL=OFF \
  -DUSE_FBGEMM=OFF -DCOMPILE_CPU=ON -DCMAKE_BUILD_TYPE=Release \
  -DCOMPILE_TESTS=ON -DUSE_MKL=OFF -DUSE_TCMALLOC=OFF -DUSE_DOXYGEN=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DBUILD_ARCH=native
cmake --build /var/lib/jenkins/moat/agent_space/marian-build-gfx1100 -j$(nproc)
```

Result: 292/292 targets, no errors. Fork clone `git status` clean -- zero build
artifacts in the repo tree.

### gfx1100 code-object evidence

```
llvm-objdump --offloading .../marian
# Extracts: marian.*.hipv4-amdgcn-amd-amdhsa--gfx1100  (11 HIP bundles)
# No gfx90a bundle present.
```

All 11 GPU code objects target gfx1100 exclusively. Zero non-gfx1100 device code.

### Wave-size audit

marian-dev's GPU kernels are wave-agnostic. Specifically:

- `reduce_all.h`: uses `cg::tiled_partition<32>` with `cg::sync(cta)` fencing
  the `tid+32` fold before entering the `thread_rank()<32` block; `shfl_down`
  runs within the 32-wide tile. On wave32 the tile == the full wavefront; no
  lockstep assumption below 32 lanes. Correct on both wave32 and wave64.
- `topk.cu` / `nth_element.cu`: the wave64 race (UNROLL_MAXARG warp-sync tail)
  was fixed in the gfx90a lead by replacing the unrolled tail with a
  `__syncthreads()`-synchronized tree to `s>0`. This fix is strictly wave-agnostic
  (block-wide barrier, no sub-wavefront assumption), correct on wave32 and wave64.
- No `__shfl` / `__ballot` / `warpSize` / `WARP_SIZE` / lane-mask in any added
  kernel line or in the marian GPU tensor/translator sources. The `warpSize`
  occurrences in `lsh_tmp.h` are a CPU template variable (starts at 4); the
  `WARP_SIZE` in `src/3rd_party/nccl/` is not compiled (USE_NCCL=OFF).
- No hardcoded launch-grid warp constant feeding a device template argument.
  No host/device WARP_SIZE split (the raft/libSGM class of bug is absent).

Wave32 verdict: kernels are fully wave-agnostic. No wave32-specific hazard found.

### Unit suites

```
BUILD=/var/lib/jenkins/moat/agent_space/marian-build-gfx1100
export HIP_VISIBLE_DEVICES=0 ROCM_PATH=/opt/rocm
$BUILD/src/tests/units/run_graph_tests
$BUILD/src/tests/units/run_attention_tests
$BUILD/src/tests/units/run_transformer_tests
$BUILD/src/tests/units/run_operator_tests
$BUILD/src/tests/units/run_rnn_tests
```

Results vs gfx90a@25f910c:
- graph_tests: 10/10 assertions, 4 test cases -- PASS (matches gfx90a)
- attention_tests: 6/6 assertions, 3 test cases -- PASS (matches gfx90a)
- transformer_tests: 3/3 assertions, 3 test cases -- PASS (matches gfx90a)
- operator_tests: 284/287 assertions; the 3 failures are the `csr-dot product`
  SECTION (operator_tests.cpp:539,609,610) -- deferred cuSPARSE SpMM path,
  identical to gfx90a. Dense dot product and topk operations PASS. (matches gfx90a)
- rnn_tests: 21/24 assertions; 3 failures are the hipRAND-vs-cuRAND reference
  mismatch (documented, not a port bug). (matches gfx90a)

Total pass tally matches gfx90a@25f910c exactly.

### End-to-end gate (wave32 correctness + determinism)

Trained a tiny Transformer (reverse-copy toy task, 1000 sentences, 600u) on GPU,
then beam-search decoded (beam=6) twice on GPU and once on CPU:

```
E2E=/var/lib/jenkins/moat/agent_space/marian-validate-gfx1100
BUILD=/var/lib/jenkins/moat/agent_space/marian-build-gfx1100

$BUILD/marian --type transformer \
  -t $E2E/train.src $E2E/train.tgt -m $E2E/model.npz \
  --vocabs $E2E/vocab.src.yml $E2E/vocab.tgt.yml --dim-emb 64 \
  --transformer-dim-ffn 128 --transformer-heads 2 --enc-depth 2 --dec-depth 2 \
  --after 600u --devices 0

$BUILD/marian-decoder -m $E2E/model.npz -v $E2E/vocab.src.yml $E2E/vocab.tgt.yml \
  -i $E2E/test.src -b 6 --devices 0 > $E2E/gpu1.out
$BUILD/marian-decoder -m $E2E/model.npz -v $E2E/vocab.src.yml $E2E/vocab.tgt.yml \
  -i $E2E/test.src -b 6 --devices 0 > $E2E/gpu2.out
$BUILD/marian-decoder -m $E2E/model.npz -v $E2E/vocab.src.yml $E2E/vocab.tgt.yml \
  -i $E2E/test.src -b 6 --cpu-threads 1 > $E2E/cpu.out

diff $E2E/gpu1.out $E2E/gpu2.out  # IDENTICAL (deterministic)
diff $E2E/gpu1.out $E2E/cpu.out   # IDENTICAL (GPU == CPU)
```

GPU run1 == GPU run2 (bit-identical, deterministic). GPU == CPU (correct).
No HSA fault (0x1016), no NaN, no hang.

gMaxElement/gMaxElementUpdate confirmed dispatching on gfx1100 via AMD_LOG_LEVEL=3:
```
ShaderName : void marian::gMaxElement<float>(...)
ShaderName : void marian::gMaxElementUpdate<float>(...)
```
The wave32-corrected topk/nth_element path is confirmed running and producing
correct beam-search results on gfx1100.

### Fork clone hygiene

`git status` in projects/marian-dev/src: clean. No build artifacts leaked into
the fork clone tree. Scratch build dir is agent_space/marian-build-gfx1100.
No fork push (validate-first follower: no source delta needed).

### Verdict: PASS -- linux-gfx1100 completed at validated_sha 25f910c

## Validation 2026-06-08 (linux-gfx90a revalidate carry-forward)

Platform: linux-gfx90a (AMD Instinct MI250X gfx90a), ROCm 7.2.1.
Delta: validated_sha 25f910ceef -> head_sha dc5cd4e2364 ([ROCm] Fix Windows Clang build for HIP port).

Changed files: CMakeLists.txt, src/3rd_party/CMakeLists.txt, src/3rd_party/sentencepiece (submodule pointer).
All changes are WIN32-guarded (Shlwapi.lib, -Wno-string-plus-int/-Wno-unused-private-field, UNICODE/CRT flags, -fPIC conditional).
One Clang/Linux-visible change: the `-march=native` exclusion for Clang compilers was removed, so `-march=native` now also applies to the Clang host compiler on Linux. This affects only host CPU code; the GPU device code is unchanged.

Built both SHAs for gfx90a in separate build dirs; ran `utils/codeobj_diff.py` on marian, marian-decoder, and run_operator_tests:
- marian: identical (56 exported symbols, device ISA identical)
- marian-decoder: identical (55 exported symbols, device ISA identical)
- run_operator_tests: identical (34 exported symbols, device ISA identical)

Commands:
```
# build old SHA (25f910c) already in agent_space/marian-build
# build new SHA (dc5cd4e):
git worktree add /var/lib/jenkins/moat/agent_space/marian-src-new dc5cd4e23649546cebc4b7cba84cf2df38b1c82d
cmake -S /var/lib/jenkins/moat/agent_space/marian-src-new \
  -B /var/lib/jenkins/moat/agent_space/marian-build-new-gfx90a -G Ninja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCOMPILE_CUDA=ON -DUSE_CUDNN=OFF -DUSE_NCCL=OFF -DUSE_FBGEMM=OFF \
  -DCOMPILE_CPU=ON -DCMAKE_BUILD_TYPE=Release -DCOMPILE_TESTS=ON \
  -DUSE_MKL=OFF -DUSE_TCMALLOC=OFF -DUSE_DOXYGEN=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DBUILD_ARCH=native
cmake --build /var/lib/jenkins/moat/agent_space/marian-build-new-gfx90a -j$(nproc)  # 292/292

python3 utils/codeobj_diff.py \
  agent_space/marian-build/marian \
  agent_space/marian-build-new-gfx90a/marian
# verdict=identical

python3 utils/codeobj_diff.py \
  agent_space/marian-build/marian-decoder \
  agent_space/marian-build-new-gfx90a/marian-decoder
# verdict=identical

python3 utils/codeobj_diff.py \
  agent_space/marian-build/src/tests/units/run_operator_tests \
  agent_space/marian-build-new-gfx90a/src/tests/units/run_operator_tests
# verdict=identical
```

Verdict: CARRY FORWARD -- linux-gfx90a completed at dc5cd4e23649546cebc4b7cba84cf2df38b1c82d (binary-equiv, no GPU re-run needed).

## Validation 2026-06-07 (windows-gfx1201, RX 9070 XT gfx1201, RDNA4) -- FAILED

Platform: AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32), Windows 11 Pro.
Only GPU on host (gfx1101 offline). HIP_VISIBLE_DEVICES=0.
Compiler: clang++ 23.0.0 (TheRock ROCm 7.14.0a20260604), cmake 3.31.
Fork: AMD-Ecosystem/marian-dev @ moat-port, head dc5cd4e (Windows build fixes on top of 25f910c).

### Build

Eight Windows-specific fixes were required to build with all-clang on Windows
(clang++ --target=x86_64-pc-windows-msvc, not clang-cl). Committed as dc5cd4e
on top of the existing port commit 25f910c. All changes are WIN32-guarded or
CMAKE_CXX_COMPILER_ID=="Clang" + WIN32; the Linux/CUDA paths are byte-identical.

Build command:
```
ROCM=/b/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel
cmake -S projects/marian-dev/src -B agent_space/marian-build-gfx1201 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_C_COMPILER="$ROCM/lib/llvm/bin/clang.exe" \
  -DCMAKE_CXX_COMPILER="$ROCM/lib/llvm/bin/clang++.exe" \
  -DCMAKE_HIP_COMPILER="$ROCM/lib/llvm/bin/clang++.exe" \
  -DCOMPILE_CUDA=ON -DUSE_CUDNN=OFF -DUSE_NCCL=OFF \
  -DUSE_FBGEMM=OFF -DCOMPILE_CPU=ON \
  -DCOMPILE_TESTS=ON -DUSE_MKL=OFF -DUSE_TCMALLOC=OFF -DUSE_DOXYGEN=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DBUILD_ARCH=native \
  -DCMAKE_PREFIX_PATH="$ROCM" -DROCM_PATH="$ROCM" \
  -DOpenBLAS_LIBRARIES="$ROCM/lib/host-math/lib/rocm-openblas.lib" \
  -DOpenBLAS_INCLUDE_DIRS="$ROCM/lib/host-math/include/openblas"
cmake --build agent_space/marian-build-gfx1201 -j32
```

Result: build succeeded, 20 link targets (marian.exe, marian-decoder.exe, etc. plus test executables).

Windows-specific build fixes committed in dc5cd4e:
1. Remove -fPIC on Windows (WIN32_UNICODE_FLAGS and FPIC_FLAG conditional)
2. Add UNICODE/_UNICODE/_DLL/_MT flags + --dependent-lib=msvcrt for CRT and wide API
3. Remove Clang exclusion from -march=native (Clang supports it fine)
4. Add Shlwapi.lib in else(MSVC) block (pathie-cpp needs PathMatchSpecW)
5. -Wno-string-plus-int / -Wno-unused-private-field to ALL_WARNINGS for WIN32+Clang
6. -Wno-string-plus-int to MARIAN_HIP_NO_WARN (same spdlog warning in HIP TUs)
7. src/3rd_party/CMakeLists.txt: guard -fPIC on libyaml-cpp/pathie-cpp with NOT WIN32
8. sentencepiece: add NOT WIN32 to the if(NOT MSVC) -fPIC guard; constexpr -> const for kAnyType

### Test results

Runtime env:
```
HIP_VISIBLE_DEVICES=0
ROCBLAS_TENSILE_LIBPATH=<_rocm_sdk_libraries>/bin/rocblas/library
HIPBLASLT_TENSILE_LIBPATH=<_rocm_sdk_libraries>/bin/hipblaslt/library
PATH=<test_dir>:<_rocm_sdk_libraries>/bin:<_rocm_sdk_core>/bin:$PATH
```

- graph_tests: 10/10 PASS -- GPU kernel dispatch confirmed on gfx1201
- binary_tests: 9/9 PASS
- utils_tests: 8/8 PASS
- fastopt_tests: 23/23 PASS
- operator_tests: 84/87 assertions PASS; 3 failures:
  - 2 failures: csr-dot product wrong values (same cuSPARSE deferred issue as Linux)
  - 1 FATAL (SIGSEGV): "affine transformation" section -- see blocker below
- attention_tests: SIGSEGV immediately (hipBLASLt crash, see blocker)
- transformer_tests: SIGSEGV immediately (hipBLASLt crash, see blocker)
- rnn_tests: ABORT "Broken type float16" -- pre-existing Windows limitation in marian's
  own code (types.h DISPATCH_BY_TYPE stubs out float16 under _MSC_VER; clang targeting
  x86_64-pc-windows-msvc defines _MSC_VER=1944); not a port bug

### Blocking issue: hipblasLtMatmulAlgoGetHeuristic crashes in libhipblaslt.dll

marian's Affine() function calls hipblasLtMatmulAlgoGetHeuristic for fused GEMM+bias
(the cublasLt epilogue path; CUDA_VERSION >= 11000 code path). On Windows with the
TheRock 7.14.0a build, libhipblaslt.dll crashes inside hipblaslt_f8::is_inf when
hipblasLtMatmulAlgoGetHeuristic is called. This is a bug in the TheRock Windows
libhipblaslt.dll, NOT a port bug.

Crash stack:
```
#0 hipblaslt_f8::is_inf (libhipblaslt.dll)  <- crash here
...
#8 string_to_epilogue_type_assert (libhipblaslt.dll)
#9 string_to_epilogue_type_assert (libhipblaslt.dll)
#10 hipblasLtMatmulAlgoGetHeuristic (libhipblaslt.dll)
#11 [run_operator_tests.exe cublasLtAffineHelper -> cublasLtMatmulAlgoGetHeuristic]
```

This crashes in both FP32 and FP16 paths, for ALL Affine calls. The crash is inside
Tensile/hipBLASLt's FP8 type-check logic even when called with FP32 data types -- a
TheRock build bug.

Affected tests: operator_tests "affine transformation" section, attention_tests,
transformer_tests. The end-to-end train/decode also cannot run (uses Affine).

Workaround for the porter: add a Windows-specific code path in prod.cpp that bypasses
hipBLASLt for affine (use hipblasGemmEx + manual BiasAdd on Windows), falling back to
the hipBLASLt path on Linux. The standalone hipblasGemmEx+hipblasSgemm work correctly
on this TheRock build; only the hipblasLt matmul descriptor API crashes.

### Verdict: validation-failed -- windows-gfx1201

GPU dispatch confirmed (graph_tests 10/10). Blocking issue: TheRock Windows
libhipblaslt.dll crashes in hipblasLtMatmulAlgoGetHeuristic for all Affine calls.
Porter must add a Windows hipBLASLt bypass (hipblasGemmEx + BiasAdd) for the
affine/attention/transformer paths.

## Validation 2026-06-08 (linux-gfx1100 revalidate carry-forward)

Platform: linux-gfx1100 (AMD Radeon Pro W7800 gfx1100), ROCm 7.2.1.
Delta: validated_sha 25f910ceef -> head_sha dc5cd4e2364 ([ROCm] Fix Windows Clang build for HIP port).

Changed files: CMakeLists.txt, src/3rd_party/CMakeLists.txt, src/3rd_party/sentencepiece (submodule pointer).
All significant changes are WIN32-guarded (Shlwapi.lib, UNICODE/CRT flags, -fPIC conditional, -Wno-string-plus-int/-Wno-unused-private-field).
Linux-visible changes: removal of `NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang"` exclusion (so -march=native now also applies to Clang host compiler on Linux) and -Wno-string-plus-int added to MARIAN_HIP_NO_WARN. Both affect only host/warning flags; GPU device code is unchanged.

Built both SHAs for gfx1100 in separate build dirs; ran `utils/codeobj_diff.py` on marian, marian-decoder, and run_operator_tests:
- marian: identical (56 exported symbols, device ISA identical)
- marian-decoder: identical (55 exported symbols, device ISA identical)
- run_operator_tests: identical (34 exported symbols, device ISA identical)

Commands:
```
# Old build at 25f910c already in agent_space/marian-build-gfx1100
# New worktree at dc5cd4e:
git worktree add /var/lib/jenkins/moat/agent_space/marian-src-new dc5cd4e23649546cebc4b7cba84cf2df38b1c82d
cp -r projects/marian-dev/src/src/3rd_party/sentencepiece/. agent_space/marian-src-new/src/3rd_party/sentencepiece/

utils/timeit.sh marian-dev compile -- cmake -S agent_space/marian-src-new \
  -B agent_space/marian-build-new-gfx1100 -G Ninja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCOMPILE_CUDA=ON -DUSE_CUDNN=OFF -DUSE_NCCL=OFF \
  -DUSE_FBGEMM=OFF -DCOMPILE_CPU=ON -DCMAKE_BUILD_TYPE=Release \
  -DCOMPILE_TESTS=ON -DUSE_MKL=OFF -DUSE_TCMALLOC=OFF -DUSE_DOXYGEN=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DBUILD_ARCH=native
cmake --build agent_space/marian-build-new-gfx1100 -j$(nproc)  # 292/292

python3 utils/codeobj_diff.py \
  agent_space/marian-build-gfx1100/marian \
  agent_space/marian-build-new-gfx1100/marian
# verdict=identical

python3 utils/codeobj_diff.py \
  agent_space/marian-build-gfx1100/marian-decoder \
  agent_space/marian-build-new-gfx1100/marian-decoder
# verdict=identical

python3 utils/codeobj_diff.py \
  agent_space/marian-build-gfx1100/src/tests/units/run_operator_tests \
  agent_space/marian-build-new-gfx1100/src/tests/units/run_operator_tests
# verdict=identical
```

Verdict: CARRY FORWARD -- linux-gfx1100 completed at dc5cd4e23649546cebc4b7cba84cf2df38b1c82d (binary-equiv, no GPU re-run needed).

## Status fix 2026-06-02: windows-gfx1151 invalid state

windows-gfx1151 was set to `blocked-needs-gfx1100`, which is not a valid
MOAT state. validate_status() rejected the file, so next_task() silently
caught the ValueError and skipped the whole project -- starving the gfx1100
port-ready follower (it could never be selected). gfx90a is completed, so
the correct follower state is `port-ready` (windowsgfx1151 gates on the
gfx90a lead like every follower, not on gfx1100). Set windows-gfx1151 ->
port-ready. (All 83 other projects use valid windows states; this was a
one-off typo from the marian-dev port.)

## Validation 2026-06-08 (windows-gfx1201, RX 9070 XT gfx1201, RDNA4) -- PASS

Platform: AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32), Windows 11 Pro.
HIP_VISIBLE_DEVICES=0 (only GPU on host). gcnArchName=gfx1201 (verified via
TheRock hipInfo.exe). Compiler: TheRock all-clang ROCm 7.14.0a20260604.
Fork AMD-Ecosystem/marian-dev @ moat-port, head 2387377 ([ROCm] Bypass hipBLASLt
for Affine on Windows ROCm), one new commit on top of dc5cd4e.

### The fix (porter)

prod.cpp: on `_WIN32 && USE_HIP`, `affineTyped<T>` no longer drives the
hipBLASLt fused GEMM+bias matmul-descriptor path; it computes the matmul with
the plain GEMM (`ProdTyped<T,T>` -> hipblasGemmEx/hipblasSgemm) then adds the
broadcast bias with the existing `BiasAdd` kernel (same optional ReLU
epilogue). This is exactly the pre-CUDA-11 fallback already in the file, so the
result is numerically identical. The post-call misalignment `BiasAdd` in
`Affine()` is guarded off on Windows+ROCm (the new affineTyped already applies
the bias). Linux/CUDA paths are byte-identical (every change is _WIN32 &&
USE_HIP guarded).

### IMPORTANT runtime requirement: ROCBLAS_USE_HIPBLASLT=0

The code bypass is necessary but on this exact TheRock 7.14 build it is not
sufficient on its own. On the current venv libraries, `rocblas_gemm_ex`
(reached by hipblasGemmEx, i.e. ALL Prod/ProdBatched/dot/bdot GEMMs) internally
delegates to `hipblaslt_ext::GemmInstance::algoGetHeuristic`, which hits the
SAME `hipblaslt_f8::is_inf` Tensile FP8 crash. So even plain GEMM SIGSEGVs
unless rocBLAS's hipBLASLt backend is disabled with `ROCBLAS_USE_HIPBLASLT=0`.
With that env set, rocBLAS uses its own Tensile kernels and every GEMM works.

Note this is a regression vs the 2026-06-07 run: that run reported the plain
`dot product` (operator_tests.cpp:508) PASSING and only Affine crashing. The
TheRock libraries in the venv have since been updated so the FP8 `is_inf` bug
now also fires through rocblas_gemm_ex. Confirmed by rebuilding the UNMODIFIED
dc5cd4e prod.cpp: the baseline also crashes at line 508 in
hipblasGemmEx->rocblas_gemm_ex->algoGetHeuristic->is_inf. This is a TheRock
runtime-library bug, not a marian/port defect; `ROCBLAS_USE_HIPBLASLT=0` is the
runtime workaround and the code bypass is the port fix. Both are required on
this host. (Not baked into marian source -- it is a TheRock library env knob.)

### Build (incremental, only prod.cpp recompiled)

```
ROCM=/b/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel
HIP_VISIBLE_DEVICES=0 utils/timeit.sh marian-dev compile -- \
  cmake --build agent_space/marian-build-gfx1201 -j24
```
(Initial configure as in the 2026-06-07 section.) Result: 23/23 targets, clean.

### Runtime env (all test/train/decode runs)

```
SP=/b/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages
export HIP_VISIBLE_DEVICES=0
export ROCBLAS_USE_HIPBLASLT=0                         # <-- the TheRock workaround
export ROCBLAS_TENSILE_LIBPATH="$SP/_rocm_sdk_libraries/bin/rocblas/library"
export HIPBLASLT_TENSILE_LIBPATH="$SP/_rocm_sdk_libraries/bin/hipblaslt/library"
export PATH="<testdir>:$SP/_rocm_sdk_libraries/bin:$SP/_rocm_sdk_core/bin:$SP/_rocm_sdk_devel/bin:$PATH"
```
PATH must be colon-separated (Git Bash); rocm-openblas.dll lives in
_rocm_sdk_core/bin and _rocm_sdk_devel/bin (needed or exit 127 "cannot open
shared object file").

### Test results (before vs after)

The previously-crashing affine/attention/transformer GPU tests now PASS (no
SIGSEGV); the previously-passing suites still pass.

```
run_operator_tests.exe "Expression graph supports basic math operations (gpu)"
  -> 202 assertions, 200 passed, 2 failed
  -> the "affine transformation" SECTION (618) PASSES (assertions 646-658),
     was a FATAL SIGSEGV before
  -> the dense "dot product" SECTION (508) PASSES, was SIGSEGV in the current
     baseline (rocblas FP8 regression) before ROCBLAS_USE_HIPBLASLT=0
  -> the 2 failures are csr-dot product (609/610), the documented deferred
     cuSPARSE path (also fails on linux-gfx90a); does NOT gate

run_attention_tests.exe "*Attention (gpu)"      -> 2/2 PASS (was SIGSEGV)
run_transformer_tests.exe "...(gpu)"            -> 1/1 PASS (was SIGSEGV)

run_graph_tests.exe    -> 10/10 PASS (unchanged)
run_binary_tests.exe   ->  9/9  PASS (unchanged)
run_utils_tests.exe    ->  8/8  PASS (unchanged)
run_fastopt_tests.exe  -> 23/23 PASS (unchanged)
```

The `(gpu, fp16)` test cases still ABORT "Broken type float16" -- the
pre-existing Windows float16 limitation (types.h stubs float16 under _MSC_VER;
clang targeting x86_64-pc-windows-msvc defines _MSC_VER). Not a port bug, not
introduced here, does not gate. Run only the float32 `(gpu)` cases.

### End-to-end (Affine in practice)

Trained a tiny Transformer (reverse-copy toy, 2000 sentences, 300u, FFN uses
Affine in fwd+bwd) on gfx1201, then beam-search decoded (beam=6) twice:
```
marian.exe --type transformer -t train.src train.tgt -m model.npz \
  --vocabs vocab.src.yml vocab.tgt.yml --dim-emb 64 --transformer-dim-ffn 128 \
  --transformer-heads 2 --enc-depth 2 --dec-depth 2 --after 300u --devices 0 \
  --shuffle-in-ram --tempdir <tmp>
marian-decoder.exe -m model.npz -v vocab.src.yml vocab.tgt.yml -i test.src -b 6 --devices 0
```
Training converged (cost 2.03 -> 1.65). Decode produced sensible reverse-copy
output and was bit-identical run-to-run (deterministic). No SIGSEGV, no NaN.
(`--shuffle-in-ram --tempdir` needed: marian's default temp-file shuffle hits a
Windows TemporaryFile::MakeTemp abort unrelated to GPU.)

### Verdict: PASS -- windows-gfx1201 completed at 2387377

The affine/attention/transformer SIGSEGVs are gone. Remaining non-gating
failures (csr-dot cuSPARSE, fp16 Windows limitation) are pre-existing and
documented. Requires runtime env ROCBLAS_USE_HIPBLASLT=0 on this TheRock build.

## Label scrub 2026-06-16 (comment-only, head 23873773 -> 401afd9f)
Dropped internal project labels from the ROCm/HIP shim comments
(src/tensors/gpu/cuda_to_hip.h + hip_compat/*.h) and the two CMakeLists.txt
ROCm-branch lines (the `# ROCm/HIP path ...` comment and the
`message(STATUS "... HIP port: CMAKE_HIP_ARCHITECTURES=...")` status string).
11 files, 12 lines, comment/message text only; no code logic touched, no
rebuild, no tests. Committed ON TOP of the validated 23873773 (no amend) as
401afd9f, pushed to AMD-Ecosystem/marian-dev moat-port (fast-forward).

advance-head classified the delta as NON-inert: linux-gfx90a, linux-gfx1100,
and windows-gfx1201 flipped completed -> revalidate (validated_sha stays
23873773, a reachable ancestor). Note the changed-line set includes a CMake
message(STATUS ...) string, which advance_head does not treat as a pure
comment, hence revalidation rather than carry-forward. No GPU re-run performed
in this scrub task. Revalidation is cheap here: a codeobj_diff binary-equiv
check (per the revalidate path) should confirm identical device ISA + exported
symbols vs 23873773 and carry forward, since the GPU code object is unchanged
by a comment/status-string edit.

## Validation 2026-06-16 (linux-gfx90a revalidate carry-forward)

Platform: linux-gfx90a (AMD Instinct MI250X gfx90a), ROCm 7.2.1.
Delta: validated_sha 23873773 -> head_sha 5367d5b7 (two comment-only commits):
- 401afd9f: Dropped "MOAT" prefix from comments in 11 files (hip_compat/*.h,
  cuda_to_hip.h, CMakeLists.txt) and changed message(STATUS "MOAT HIP port: ...")
  to message(STATUS "HIP port: ...").
- 5367d5b7: Replaced "byte-identical" with "unchanged" in 3 files (CMakeLists.txt,
  cuda_to_hip.h, prod.cpp comments).

All changes are comment/string text only. No logic, no GPU kernel code, no device
headers, no CMake variable or build-flag changes. The message(STATUS ...) change
is a configure-time print only (does not affect code generation).

Built both SHAs for gfx90a in separate build dirs using sentencepiece 1ca221c
(the public marian-nmt fork HEAD; the submodule pointer f006008f is unreachable
from the public GitHub API -- local build workaround, not committed):

```
# Configured both build dirs, then:
cmake --build agent_space/marian-build-old-gfx90a -j$(nproc)  # 188/188 targets
cmake --build agent_space/marian-build-new-gfx90a -j$(nproc)  # 84/84 targets
```

Ran utils/codeobj_diff.py on 7 executables:
- marian: identical (56 exports, device ISA identical)
- marian-decoder: identical (55 exports, device ISA identical)
- run_operator_tests: identical (34 exports, device ISA identical)
- run_attention_tests: identical (34 exports, device ISA identical)
- run_transformer_tests: identical (34 exports, device ISA identical)
- run_graph_tests: identical (29 exports, device ISA identical)
- run_rnn_tests: identical (34 exports, device ISA identical)

## Validation 2026-06-16 (windows-gfx1201 revalidate carry-forward)

Platform: windows-gfx1201 (AMD Radeon RX 9070 XT gfx1201, RDNA4), Windows 11 Pro.
GPU verified: HIP_VISIBLE_DEVICES=1 -> AMD Radeon RX 9070 XT.
Delta: validated_sha 23873773 -> head_sha 5367d5b (two comment-only commits):
- 401afd9f: Dropped "MOAT" prefix from comments in 11 files (hip_compat/*.h,
  cuda_to_hip.h, CMakeLists.txt) and changed message(STATUS "MOAT HIP port: ...")
  to message(STATUS "HIP port: ...").
- 5367d5b7: Replaced "byte-identical" with "unchanged" in 3 files (CMakeLists.txt,
  cuda_to_hip.h, prod.cpp comments).

Built HEAD (5367d5b) for gfx1201 into agent_space/marian-build-gfx1201-new (292/292
targets). The validated_sha build (2387377) is still in agent_space/marian-build-gfx1201.

Binary-equivalence method for Windows PE (ELF-based codeobj_diff.py does not run on
Windows PE binaries): extracted .hip_fat PE sections with llvm-objcopy, parsed the
clang offload bundles inside, and compared the individual device ELF .text sections
(actual gfx1201 ISA) byte by byte.

Results for marian.exe (11 ELF objects in .hip_fat):
- All 11 .text sections: byte-identical
- The only difference is __hip_cuid_662cf5b03a3228fa (old) vs __hip_cuid_24a98afdb79321cc (new)
  in the .dynsym/.dynstr/.strtab of each ELF -- this is a build-time source-content hash
  embedded as a symbol name by the HIP runtime registration glue, not device ISA code.
- Exported PE symbols: identical (24 exports, unchanged)
- Section sizes: identical

Same result for run_operator_tests.exe and other GPU executables (same 11 ELF objects,
same .text identical / same __hip_cuid_ symbol-name-only diff).

Verdict: CARRY FORWARD -- windows-gfx1201 completed at 5367d5b704ef5db827af5cac2a01efebb0499939
(binary-equiv, no GPU re-run needed). Device ISA unchanged; only __hip_cuid_ symbol names
differ due to source-content hash recomputation from the comment changes.

Verdict: CARRY FORWARD -- linux-gfx90a completed at 5367d5b704ef5db827af5cac2a01efebb0499939 (binary-equiv, no GPU re-run needed).

## Validation 2026-06-17 (linux-gfx1100 revalidate carry-forward)

Platform: linux-gfx1100 (AMD Radeon Pro W7800 gfx1100), ROCm 7.2.1.
Delta: validated_sha 23873773 -> head_sha 5367d5b7 (two comment-only commits):
- 401afd9f: Dropped "MOAT" prefix from comments in 11 files (hip_compat/*.h,
  cuda_to_hip.h, CMakeLists.txt) and changed message(STATUS "MOAT HIP port: ...")
  to message(STATUS "HIP port: ...").
- 5367d5b7: Replaced "byte-identical" with "unchanged" in 3 files (CMakeLists.txt,
  cuda_to_hip.h, prod.cpp comments).

moatlib classify verdict: mixed (CMakeLists.txt flagged as literal-token change due
to the message(STATUS) string; all other files comment-only). The STATUS string is
a configure-time print only; no code logic or device code is affected.

Built both SHAs for gfx1100 in separate build dirs (HIP_VISIBLE_DEVICES=1):

```
# Old SHA (23873773) worktree:
git worktree add agent_space/marian-src-old-23873773 23873773a4e90c74935c8ead05af7f9a7f917ee7
cp -r projects/marian-dev/src/src/3rd_party/sentencepiece/. agent_space/marian-src-old-23873773/src/3rd_party/sentencepiece/
cmake -S agent_space/marian-src-old-23873773 -B agent_space/marian-build-old-23873773-gfx1100 -G Ninja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCOMPILE_CUDA=ON -DUSE_CUDNN=OFF -DUSE_NCCL=OFF \
  -DUSE_FBGEMM=OFF -DCOMPILE_CPU=ON -DCMAKE_BUILD_TYPE=Release \
  -DCOMPILE_TESTS=ON -DUSE_MKL=OFF -DUSE_TCMALLOC=OFF -DUSE_DOXYGEN=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DBUILD_ARCH=native
cmake --build agent_space/marian-build-old-23873773-gfx1100 -j$(nproc)  # 292/292

# New SHA (5367d5b7) -- main clone at HEAD:
cmake -S projects/marian-dev/src -B agent_space/marian-build-new-5367d5b7-gfx1100 -G Ninja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  [same flags as above]
cmake --build agent_space/marian-build-new-5367d5b7-gfx1100 -j$(nproc)  # 292/292
```

Ran utils/codeobj_diff.py on 5 GPU executables:
- marian: identical (56 exports, device ISA identical)
- marian-decoder: identical (55 exports, device ISA identical)
- run_operator_tests: identical (34 exports, device ISA identical)
- run_attention_tests: identical (34 exports, device ISA identical)
- run_transformer_tests: identical (34 exports, device ISA identical)

Verdict: CARRY FORWARD -- linux-gfx1100 completed at 5367d5b704ef5db827af5cac2a01efebb0499939 (binary-equiv, no GPU re-run needed). Device ISA unchanged; the only delta is a configure-time STATUS string and comment text.

## Validation 2026-06-19 (windows-gfx1101, AMD Radeon PRO V710, gfx1101, RDNA3) -- FAILED

Platform: AMD Radeon PRO V710 (gfx1101, RDNA3, wave32), Windows 11 Pro.
HIP_VISIBLE_DEVICES=1 (gfx1101; mask 0 = gfx1201 RX 9070 XT).
GPU verified: hipInfo shows AMD Radeon PRO V710 / gcnArchName=gfx1101.
Compiler: TheRock all-clang ROCm 7.14.0a20260604.
Fork AMD-Ecosystem/marian-dev @ moat-port, head 5367d5b (same as all other completed platforms).

### Build

Build for gfx1101 from scratch in agent_space/marian-build-gfx1101:

```
SP=/b/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages
ROCM=$SP/_rocm_sdk_devel
cmake -S projects/marian-dev/src -B agent_space/marian-build-gfx1101 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1101 \
  -DCMAKE_C_COMPILER=$ROCM/lib/llvm/bin/clang.exe \
  -DCMAKE_CXX_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
  -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++.exe \
  -DCOMPILE_CUDA=ON -DUSE_CUDNN=OFF -DUSE_NCCL=OFF \
  -DUSE_FBGEMM=OFF -DCOMPILE_CPU=ON \
  -DCOMPILE_TESTS=ON -DUSE_MKL=OFF -DUSE_TCMALLOC=OFF -DUSE_DOXYGEN=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DBUILD_ARCH=native \
  -DCMAKE_PREFIX_PATH=$ROCM -DROCM_PATH=$ROCM \
  -DOpenBLAS_LIBRARIES=$ROCM/lib/host-math/lib/rocm-openblas.lib \
  -DOpenBLAS_INCLUDE_DIRS=$ROCM/lib/host-math/include/openblas
cmake --build agent_space/marian-build-gfx1101 -j64
```

Result: 292/292 targets, no errors.

### Runtime env

```
export HIP_VISIBLE_DEVICES=1
export ROCBLAS_USE_HIPBLASLT=0
export ROCBLAS_TENSILE_LIBPATH=$SP/_rocm_sdk_libraries/bin/rocblas/library
export HIPBLASLT_TENSILE_LIBPATH=$SP/_rocm_sdk_libraries/bin/hipblaslt/library
export PATH=<test_dir>:$SP/_rocm_sdk_libraries/bin:$SP/_rocm_sdk_core/bin:$SP/_rocm_sdk_devel/bin:$PATH
```

TheRock runtime DLLs (amdhip64_7.dll, amd_comgr.dll, rocm_kpack.dll, hiprtc*.dll, hipblas.dll, hiprand.dll, hipsparse.dll, rocblas.dll, libhipblaslt.dll, rocm-openblas.dll) copied next to all executables. rocblas/library/ also copied alongside each exe dir.

### Unit test results

- run_binary_tests.exe: 9/9 PASS
- run_utils_tests.exe: 8/8 PASS
- run_fastopt_tests.exe: 23/23 PASS
- run_graph_tests.exe: 10/10 PASS -- GPU dispatch confirmed on gfx1101
- run_attention_tests.exe "*Attention (gpu)": 2/2 PASS (was SIGSEGV on first gfx1201 attempt; fixed by ROCBLAS_USE_HIPBLASLT=0)
- run_transformer_tests.exe "*(gpu)": 1/1 PASS
- run_operator_tests.exe "Expression graph supports basic math operations (gpu)":
  200/202 assertions PASS, 2 failures = csr-dot product (same deferred cuSPARSE issue as all other platforms)
  Affine transformation section PASSES. bdot (batched GEMM unit test) PASSES.
- run_rnn_tests.exe: ABORT "Broken type float16" (pre-existing Windows limitation, same as gfx1201).

### Training smoke (GPU)

```
marian.exe --type transformer -t train.src train.tgt -m model.npz \
  --vocabs vocab.src.yml vocab.tgt.yml --dim-emb 64 --transformer-dim-ffn 128 \
  --transformer-heads 2 --enc-depth 2 --dec-depth 2 --after 300u --devices 0 \
  --shuffle-in-ram --tempdir tmp
```

Training completed successfully (300u). Model checkpoint saved. GPU used for all forward+backward passes (Affine, ProdBatched via hipBLAS single GEMM path). No SIGSEGV, no crash.

### Blocking issue: rocBLAS grouped-batched GEMM (ISA000) fails on Windows gfx1101

marian-decoder.exe uses ProdBatched -> hipblasGemmBatchedEx (pointer-array batched GEMM) for beam-search decode. For the matrix sizes used in decode (larger than the unit-test bdot case), rocBLAS selects the `Cijk_Alik_Bljk_SB_GB_MT128x64x12_SN_1LDSB0_...ISA000_...` kernel -- a generic SPIR-V ("ISA000") grouped-batched GEMM kernel. This kernel is NOT available in the per-arch Tensile library (the gfx1101 Tensile .dat files don't contain any GB-variant solutions); rocBLAS falls back to the generic SPIR-V path embedded in the DLL. On Windows, `kpack_load_code_object` fails with error 13 loading this SPIR-V code object, returning `hipErrorInvalidImage:200`.

This is a TheRock/rocBLAS Windows/gfx1101 limitation: the SPIR-V generic fallback for large grouped-batched GEMM does not work on Windows ROCm for gfx1101. The unit test `bdot` passes because it uses small matrices that hit a different (ELF/.hsaco) kernel path. The decoder decode fails because it uses larger batched GEMM shapes.

This limitation is NOT present on gfx1201: the gfx1201 grouped-batched kernels appear to be in ELF format or use a different dispatch path that avoids the SPIR-V fallback.

Error log:
```
hip_fatbin.cpp:710: kpack_load_code_object failed with error: 13
hipLaunchKernel: Returned hipErrorInvalidImage
rocBLAS error from hip error code: 'hipErrorInvalidImage':200
Error: Cublas Error: 6 -- prod.cpp:179: cublasGemmBatchedEx(...)
```

Workaround possibility: replace hipblasGemmBatchedEx with hipblasGemmStridedBatchedEx (strided layout, different Tensile dispatch) or a loop of individual hipblasGemmEx calls for the Windows+gfx1101 case. Since the batched matrices in marian's ProdBatched ARE contiguous, a strided batched call would be semantically equivalent. This would be a genuine port fix if taken.

### Verdict: validation-failed -- windows-gfx1101

GPU unit tests all pass on real gfx1101 GPU (graph/attention/transformer/operator/bdot). Training smoke passes on GPU. The e2e GPU decode is blocked by a rocBLAS SPIR-V grouped-batched GEMM limitation on Windows/gfx1101 for decoder-scale matrix sizes. This is a TheRock library limitation for this platform, not a port correctness issue. gfx1101 is OPTIONAL (never blocks PRs); windows-gfx1201 is already completed at this head.

If a fix is desired: prod.cpp needs a `_WIN32 && USE_HIP && defined(ROCBLAS_BATCHED_GEMM_WORKAROUND)` path that replaces `hipblasGemmBatchedEx` with `hipblasGemmStridedBatchedEx` (since marian's ProdBatched always has contiguous stride) or a sequential GEMM loop for Windows gfx1101.

## 2026-08-10: Sent back by the maintainer -- the three scope-outs are now in scope

Jeff Daily rejected the port's scope on review PR #1 (review 4898442640,
2026-08-10): the three deliberate scope limits were never signed off, and the
SpMM, NCCL, and cuDNN features should be ported, not deferred. The upstream PR
marian-nmt/marian-dev#1043 stays OPEN as a DRAFT (comment posted: porting
additional features, will mark ready when complete) -- do not close it, and do
not mark it ready until all three land and revalidate. Stage set back to
`porting` (lock: linux-gfx1100). What each feature needs:

1. cuSPARSE SpMM sparse-attention path (prod_sparse_cu11.h CSRProd): the
   hipSPARSE generic-API port currently fails HIPSPARSE_STATUS_NOT_SUPPORTED
   (the 3 failing operator_tests csr-dot assertions, operator_tests.cpp:539,
   609,610). Needs hipSPARSE-specific order/alg work, not name-for-name
   mapping.
2. cuDNN convolution/pooling -> MIOpen (cudnn_wrappers.cu,
   graph/node_operators_binary.h, layers/convolution.cpp): currently kept
   CUDA-only behind USE_CUDNN=OFF. Needs a real MIOpen backend so USE_CUDNN=ON
   (or a HIP equivalent) builds and the char-CNN conv/pool path runs on ROCm.
3. NCCL -> RCCL multi-GPU collectives (training/communicator_nccl.h; the
   vendored nccl submodule does not build under hipcc): swap to RCCL, which is
   API-compatible and ships in ROCm, and build with USE_NCCL=ON.

Validating feature 3 is unlike everything validated so far: collectives are a
multi-GPU feature, so the test process must hold two or more same-arch GPUs at
once, and other ports' GPU jobs cannot be parallelized onto the sibling GPUs
during those runs. First MOAT project to exercise RCCL; the scheduling rule is
in the cuda-to-rocm skill's references/validation.md.

Re-porting advances head_sha, so the completed platforms (linux-gfx90a,
linux-gfx1100, windows-gfx1201) flip to revalidate as fixes land -- expected,
do not suppress. The existing deferrals (marian-topk-cub-segsort,
marian-gfx1101-batched-gemm-spirv) are separate items, untouched by this
ruling.

## 2026-08-10: the three scope-outs ported (linux-gfx1100, porter)

All three features the maintainer sent the port back for are implemented,
built and exercised on real GPU here. Fork moat-port ba0ec806 -> 1381ed77
(five commits, none amended; ba0ec806 stays a reachable ancestor).

```
4a257f29 [ROCm] Make the sparse GEMM path work under hipSPARSE
ea834eb1 [ROCm] Fix half-precision scaling in the fused affine GEMM
bd190deb [ROCm] Build the multi-GPU collectives against RCCL
6cd25824 [ROCm] Run convolution and pooling on MIOpen
1381ed77 [ROCm] Document AMD GPU support
```

Host: 4x AMD Radeon Pro W7800 48GB (gfx1100, RDNA3, wave32), ROCm 7.2.3.

### 1. cuSPARSE SpMM -> hipSPARSE (prod_sparse_cu11.h)

Two separate causes, neither the order/alg guess the deferral assumed.
`HIPSPARSE_ORDER_ROW` with `CSR_ALG2` is in fact correct and matches a host
reference for both transS values; a standalone probe over every
{ORDER_ROW,ORDER_COL} x {ALG_DEFAULT,CSR_ALG1,CSR_ALG2,CSR_ALG3} combination
showed ORDER_ROW right in all four.

- fp32 wrong results: upstream calls SpMM only inside `if(bufferSize > 0)`.
  cuSPARSE always wants scratch for CSR_ALG2 so the shortcut never fires on
  NVIDIA; hipSPARSE reports 0 for these shapes, so the product was never
  issued and C kept its previous contents. Fixed by making the call
  unconditional and only the allocation conditional.
- fp16 HIPSPARSE_STATUS_NOT_SUPPORTED: hipSPARSE has no uniform half SpMM.
  rocsparse_spmm.h tabulates the supported combinations: half A/B are accepted
  only with a FLOAT C and float compute. The half instantiation now multiplies
  into a float scratch matrix and casts back with CopyCast, seeding the scratch
  from C when beta is non-zero. Also the pre-existing HIP workaround set the
  compute type to 32F while still passing a `half` alpha -- a four-byte read of
  a two-byte value; scalars are now float on this path.

Note `gpu::CopyCast` must be called qualified: unqualified lookup finds
`marian::gpu::CopyCast` while ADL on `marian::Tensor` also finds the
`marian::CopyCast` dispatcher, and the call is ambiguous.

### 2. The affine bug the sparse failure was hiding

With csr-dot fixed, operator_tests went from 287 assertions to 603: Catch2
had been aborting the whole test case at the csr-dot exception, so every
section after it never ran on any platform. Three of the newly reached
assertions failed at once (operator_tests.cpp:648,653,658, "affine
transformation", fp16 GPU case only): `affine()` returned +/-inf and
`affineWithReluDropout()` returned the bias alone.

Cause: the hipBLASLt matmul descriptor uses a 32F scale type on HIP (16F is
rejected), but the half overload of `cublasLtAffineTyped` still passed
`const half*` alpha/beta. Same four-byte-read-of-two-bytes fault as above.
Fixed by widening both scalars to float in that overload.

This is the load-bearing lesson of the session and is promoted to the skill:
a failing section's assertion count is a lower bound on what went untested.

### 3. NCCL -> RCCL (multi-GPU collectives)

`find_package(rccl)` + link `roc::rccl`; a `hip_compat/nccl.h` forwarding to
`<rccl/rccl.h>` (RCCL keeps the NCCL names, so communicator_nccl.h is
unchanged); the vendored NCCL ExternalProject is skipped under USE_HIP; and
`cudaStreamCreate`/`cudaStreamDestroy` added to the compat header.
USE_NCCL=ON is now the working default.

Validated for real on four GPUs, not compile-only -- this host has 4x gfx1100,
contrary to the single-GPU assumption in the dispatch. Log shows
`[comm] Using NCCL 4.7.7 for GPU communication` and `NCCLCommunicators
constructed successfully`; synchronous SGD over `--devices 0 1 2 3` converges
(cost 0.52 @1500u -> 0.057 @3000u) and the trained model decodes the toy
reverse-copy task correctly. Asynchronous (default) multi-GPU also converges.

### 4. cuDNN -> MIOpen (convolution and pooling)

cudnn_wrappers.{h,cu} keep one set of class declarations; the handle and
descriptor types are aliased to MIOpen's (MIOpen uses an ordinary tensor
descriptor for conv weights, not a filter type) and the MIOpen implementation
is a self-contained block ahead of the untouched cuDNN one.

Three MIOpen-vs-cuDNN differences drove the shape of it:
- Explicit workspace + an algorithm chosen by benchmarking. THE TRAP: the
  `miopenFindConvolution*Algorithm` calls WRITE to whatever tensor is named as
  their output. Passing the live `x` and kernel buffers to the backward finds
  (which the signatures accept) silently corrupted the model -- char-s2s
  training produced NaN from update 1 and a multi-output-channel conv gave
  results ~4000x too large with the right sign pattern. Each find now gets a
  destination that is about to be overwritten anyway. Cached per input shape.
- Only alpha=1/beta=0 are honoured in 2D, so the accumulating backward form is
  a conv into a staging buffer plus `miopenOpTensor(miopenTensorOpAdd,...)`.
  Bias addition likewise: `miopenConvolutionForwardBias` overwrites, so the
  forward bias is an OpTensor add.
- Pooling backward reads the indices its forward recorded, so that workspace is
  a wrapper member, not a local.

Two upstream defects in the same CUDNN-gated code had to be fixed to make it
work at all, and both affect the CUDA build identically:
- `Options` never instantiated `Get<std::pair<int,int>>`, which is how the
  convolution layer reads kernel-dims/paddings/strides. Nothing linked with
  USE_CUDNN=ON on ANY platform. Added the instantiation (yaml-cpp already has
  a `convert<std::pair>`).
- `PoolingOp` kept its input's shape instead of asking the wrapper for the
  pooled one, so any window that actually shrinks the input left most of the
  output unwritten. Now calls `getOutputShape` the way `ConvolutionOp` does.

The `pooling` app test is back in the HIP build (it is nearly all commented
out upstream and proves only that the graph constructs).

### Test results (clean rebuild, gfx1100, all three features ON)

Configure/build: `-DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100
-DUSE_CUDNN=ON -DUSE_NCCL=ON` (rest as the recipe above). 294/294 targets.

- operator_tests: 603/603 assertions, 4 test cases -- PASS (was 284/287 with
  the case aborting). fp32 GPU 202/202, fp16 GPU 202/202, CPU 198/198.
- graph 10/10, attention 6/6, transformer 3/3, binary 9/9, fastopt 23/23,
  utils 8/8 -- all PASS.
- rnn_tests 21/24 -- the three documented hipRAND-vs-cuRAND reference
  mismatches, unchanged. Not a port bug.
- App tests: logger, dropout, prod, pooling exit 0. sqlite and cli abort for
  want of command-line arguments (pre-existing, they take required args).
- Convolution/pooling correctness: a standalone program linked against the
  built library checks conv forward, d/dinput, d/dkernel, d/dbias, and max and
  average pooling forward+backward against hand-computed references, plus a
  multi-output-channel conv of the character-encoder's shape against a host
  reference. All agree. Source: agent_space/conv_pool_check.cpp (not committed
  to the fork; it is a validation aid, not a project test).
- char-s2s (the character-CNN encoder, the only consumer of the conv path)
  trains on GPU: cost 2.97 -> 1.70 over 1000 updates. Needs
  `--char-highway 0`: the highway network calls `sigmoid(vector<Expr>)`, which
  upstream ABORTs with "Not implemented" on every platform. Unrelated to ROCm,
  not fixed here.
- End-to-end determinism gate (single GPU): transformer trained to cost 0.044
  @3000u, beam-6 decode run1 == run2 AND GPU == CPU. The wave-size fix in
  topk/nth_element still holds.
- Multi-GPU: see feature 3 above.

### Gotcha: the sentencepiece submodule pin is unfetchable

`git submodule update --init src/3rd_party/sentencepiece` fails with
`upload-pack: not our ref f006008f97c8a724d2dee306fb5347b109dbb893` -- the
commit marian-dev pins is gone from marian-nmt/sentencepiece. Recovery: let the
clone land, then `git -C src/3rd_party/sentencepiece checkout -f master`
(1ca221c, v0.1.95), which has the spm_* targets marian's CMake expects and
builds fine. This leaves the gitlink showing as modified; NEVER stage it, and
`git submodule deinit -f src/3rd_party/sentencepiece` restores a clean tree
when you are done building. Upstream problem, not ours.

### Owed / not done

- Nothing from the maintainer's three items is outstanding.
- The fp16 SpMM path adds a cast round-trip per call. Correct but not tuned;
  if sparse-attention fp16 throughput ever matters, revisit.
- gfx90a and windows-gfx1201 flip to revalidate at 1381ed77 (expected: real
  functional change). Both need USE_CUDNN=ON/USE_NCCL=ON runs; note the
  Windows arch has no MIOpen/RCCL story validated yet, and RCCL there needs
  2+ same-arch GPUs in one process.

## Review 2026-08-10 (reviewer, linux-gfx1100, fork moat-port ba0ec806..1381ed77)

Verdict: CHANGES REQUESTED. One blocker (the CUDA build no longer compiles),
two smaller code fixes, one record fix. The MIOpen, RCCL and hipSPARSE work is
otherwise sound; the detail of what was checked is at the end.

### Blocker: the CUDA build stops compiling (fp16 sparse GEMM)

`src/tensors/gpu/prod_sparse_cu11.h:191`

```cpp
if(betaScalar == (ScalarType)0)
```

On the CUDA path `ScalarType` is `ElementType` (line 38), so for the half
instantiation this is `(__half)0` -- a cast from an `int` literal.  Outside
`__CUDACC__`, older CUDA headers expose only the `float` and `double`
constructors of `__half` (the integer converts sit behind
`#if defined(__CUDACC__)`), and `int` converts to both at the same rank, so the
call is ambiguous.  `prod_sparse.cpp` is a plain host TU on the CUDA path (the
legacy `cuda_add_library` sends only `.cu` to nvcc), it defines `COMPILE_FP16`
whenever `CUDA_FOUND` is set (`common/types.h:44`), and it instantiates
`TypedSparseGemm<half>` at `prod_sparse.cpp:36`, so the branch is instantiated
even though `resultNeedsCast()` is constant-false there.

Reproduced with the exact template shape and real CUDA headers, host compiler:

```
error: call of overloaded '__half(int)' is ambiguous
   note: candidate: '__half::__half(double)'
   note: candidate: '__half::__half(float)'
```

with CUDA 12.1 headers; CUDA 12.8 headers compile it, because that release made
the integer converts visible to host compilers
(`__CUDA_FP16_DISABLE_IMPLICIT_INTEGER_CONVERTS_FOR_HOST_COMPILERS__`).  This
file exists for `CUDA_VERSION >= 11000`, so the broken range covers most of what
it is meant to serve, and 4a257f29's message ("The CUDA path is unchanged apart
from the buffer-size condition") does not hold as written.

Fix: use a float literal (`(ScalarType)0.f`) or compare in float
(`(float)betaScalar == 0.f`).  Better, make both `if(resultNeedsCast())` blocks
(lines 188 and 256) `if constexpr` -- the project is C++17, and that stops the
CUDA build from instantiating a block that can never run there at all.

Then compile-check it: a host-compile of `prod_sparse.cpp` against a CUDA
toolkit older than 12.2 is what catches this class, and neither the HIP build
nor a 12.8 host compile will.

### Pooling backward allocates and frees device memory on every call

`src/tensors/gpu/cudnn_wrappers.cu:459-460`

```cpp
DeviceBuffer staging
    = allocateDeviceBuffer(xGrad->shape().elements() * sizeof(float));
```

`allocateDeviceBuffer` is `hipMalloc`, and the `shared_ptr` deleter `hipFree`s
at scope exit, so this is a device allocation and a (synchronizing) free per
pooling node per batch on the training path.  The convolution gradients in the
same file solve the identical problem with a cached member and the `ensureBuffer`
helper (`cudnn_wrappers.cu:294`), and `PoolingWrapper` already carries the
member pattern for its index workspace.  Add a second member (`gradStaging_` /
`gradStagingSize_`) and use `ensureBuffer`.

### MIOPEN_CALL reports a bare status number

`src/tensors/gpu/cudnn_wrappers.cu:33`

`printf("Error (%d) ...", (int)_s)` where the cuDNN macro it mirrors prints
`cudnnGetErrorString(x)`.  MIOpen has the same decoder (`miopenGetErrorString`,
`miopen.h:138`); use it, so a failure in the field says what failed.

### notes.md still says the three features are deferred

`projects/marian-dev/notes.md:114-126` ("Deferred follow-ups") asserts that the
sparse path is broken, that cuDNN is CUDA-only, that RCCL is a follow-up and
that "the `pooling` app test is dropped from the HIP build".  All four are now
false, and that section is read before the 2026-08-10 one.  Mark it superseded
rather than leaving two sections that contradict each other.

### Checked and clean

- fp16 affine fix (ea834eb1): `cublasLtMatmul` has exactly one call site
  (prod.cpp:674) reached by exactly two typed overloads; the half one now widens
  and the float one already matched the 32F scale type, so the fix is complete.
  `alpha`/`beta` reach it as host pointers (`&alpha` of a local in
  `affineTyped`, prod.cpp:776), so the host-side dereference is valid, and the
  `#else` arm is character-identical to the original CUDA call.
- hipSPARSE rework (4a257f29): the mixed-precision selection matches the table
  in `rocsparse_spmm.h` (A/B f16_r, C f32_r, compute f32_r) -- descS/descD 16F,
  descC 32F, float scalars.  The beta!=0 seeding of the float scratch is the
  correct accumulate form.  Dropping the `bufferSize > 0` guard leaves cuSPARSE
  behaviour intact in every case where it previously ran, and fixes the case
  where it silently did not.
- MIOpen find calls (6cd25824): all three finds get a destination that is about
  to be overwritten -- forward takes the real `y` (overwritten by the following
  `miopenConvolutionForward` with beta=0), and both backward finds take the
  staging buffer, never the live `x`/kernel tensors.  Workspace sizing is queried
  per direction before each find, `ensureBuffer` only grows, and passing a
  capacity larger than the queried need is safe.  Pooling has no find.
- Every MIOpen signature used was checked against `/opt/rocm/include/miopen/miopen.h`
  argument by argument (`miopenSet2dPoolingDescriptor`, `miopenPoolingBackward`,
  `miopenPoolingGetWorkSpaceSizeV2`, the three convolution calls, `miopenOpTensor`),
  and the alpha=1/beta=0 restriction the port designs around is documented at
  miopen.h:1831, 1877 and 2184.
- Stream safety: `miopenCreate` leaves the handle on the NULL stream (probed on
  this host: `miopenGetStream` returns nil), which is where marian's kernels
  launch, so the MIOpen work is ordered against them exactly as cuDNN's is.  No
  hidden private-stream race.
- The two upstream defects are genuinely platform-neutral: `Get<std::pair<int,int>>`
  is required by `layers/convolution.cpp:13-16`, yaml-cpp has `convert<std::pair>`
  (`3rd_party/yaml-cpp/node/convert.h:283`) and fastopt has `As<std::pair>`, so the
  instantiation is well-formed in every configuration; `PoolingOp` (guarded by
  `#ifdef CUDNN`) now asks the wrapper for its shape the way `ConvolutionOp`
  always did.  Neither can regress a working CUDA user, since nothing linked with
  USE_CUDNN=ON before the first of them.
- RCCL: `find_package(rccl)`/`roc::rccl` and `find_package(miopen)`/`MIOpen` are
  the real ROCm package and target names (verified in /opt/rocm/lib/cmake).  The
  quoted `#include "nccl.h"` in communicator_nccl.h resolves through the compat
  dir already on the marian target's include path; `USE_NCCL` is referenced only
  by host TUs, so its absence from CMAKE_HIP_FLAGS is correct.  The vendored NCCL
  ExternalProject is skipped by `CUDA_FOUND AND NOT USE_HIP`, leaving the NVIDIA
  branch untouched.
- Re-adding the `pooling` app test to the HIP build is safe with USE_CUDNN=OFF:
  `src/tests/pooling.cpp` has its whole pooling body commented out upstream and
  names no CUDNN-gated type.
- Fault classes: no new device code at all in these five commits -- no kernel
  launch, no warp intrinsic, no literal 32, no texture or surface.  The new
  resource holders are `shared_ptr` with a `hipFree` deleter, which leaves the
  wrappers exactly as copyable as the cuDNN versions were.
- Skill lessons: both new fault-classes entries and both validation.md entries
  are true as written and correctly filed.  Spot-checked against the sources they
  cite -- the MIOpen alpha/beta restriction and the rocSPARSE mixed-precision
  table are in the installed headers, and the Catch2 section-abort behaviour
  matches what the 284->603 assertion jump shows.  The RCCL scheduling lesson
  reads correctly now that the host turned out to have four gfx1100.
- Hygiene: `jargon.py --port marian-dev` clean; five `[ROCm]` titles, longest 58
  chars; Claude named in every body, no noreply trailer, no AMD-internal account
  or hostname; ASCII throughout the diff and the messages; fork worktree clean.

### Not re-litigated

ba0ec806 and earlier, which three platforms already validated.  GPU test runs
for these five commits are the validator's job and their absence is not part of
this verdict.

## 2026-08-10: review findings fixed (linux-gfx1100, porter)

All four findings of the 2026-08-10 review are addressed. Fork moat-port
1381ed77 -> 29ec0725, three new commits, nothing amended (1381ed77 stays a
reachable ancestor).

```
32d65345 [ROCm] Keep the sparse GEMM compiling for NVIDIA
4ff41294 [ROCm] Cache the pooling backward staging buffer
29ec0725 [ROCm] Report MIOpen failures by name, not by number
```

### 1. Blocker: the CUDA build (prod_sparse_cu11.h)

Confirmed exactly as the review described, and fixed with `if constexpr` on
both `resultNeedsCast()` blocks (188, 256) plus a float literal in the beta
comparison. The reviewer's diagnosis was right down to the line: with the
pre-fix source, the REAL translation unit (not just a repro shape) fails.

Repro, marian's own prod_sparse.cpp host-compile command with the CUDA 12.1
fp16 headers ahead of the 12.8 ones (cusparse.h still comes from 12.8; only
cuda_fp16.h version matters here):

```
prod_sparse_cu11.h:191:22: error: call of overloaded '__half(int)' is ambiguous
  191 |     if(betaScalar == (ScalarType)0)
/opt/conda/envs/cuda-12.1-hdr/include/cuda_fp16.hpp:215:25: note: candidate: '__half::__half(double)'
/opt/conda/envs/cuda-12.1-hdr/include/cuda_fp16.hpp:214:25: note: candidate: '__half::__half(float)'
```

instantiated from `TypedSparseGemm<__half>::CSRProd` at prod_sparse.cpp:37.
With the fix the same command compiles clean.

CUDA 12.1 headers came from `conda create -n cuda-12.1-hdr -c nvidia
cuda-cudart-dev=12.1 cuda-cccl=12.1` (cuda-cccl is needed too, or cuda_fp16.hpp
fails on `#include <nv/target>`); CUDART_VERSION 12010, host g++ 13.3.

Full nvcc CUDA-build check per the skill: configure with `-DUSE_HIP=OFF
-DCUDA_TOOLKIT_ROOT_DIR=/opt/conda/envs/cuda-12.8/targets/x86_64-linux` and the
arch pinned to sm_80 (marian uses the legacy FindCUDA COMPILE_<arch> switches,
NOT CMAKE_CUDA_ARCHITECTURES: pass `-DCOMPILE_AMPERE=on` with KEPLER/MAXWELL/
PASCAL/VOLTA/TURING/AMPERE_RTX off). 161/161 targets, all five executables
LINKED. Compile-and-link only; no NVIDIA GPU here, nothing was run.

Two environment gotchas for the next CUDA check on a conda toolkit, both local
to this host and not port defects:
- FindCUDA wants `CUDA_TOOLKIT_ROOT_DIR` at `targets/x86_64-linux` (that is
  where `include/cuda_runtime.h` lives), but nvcc then resolves its internal
  tools relative to `targets/x86_64-linux/bin`, which holds only nvcc, and the
  build dies with `sh: 1: cudafe++: not found`. Symlink cudafe++, ptxas,
  nvlink, fatbinary, cicc, bin2c, crt and nvcc.profile into that bin. Also
  symlink `lib64 -> lib` in the env root, or marian's cublasLt find fails.
- `cumsum.cu` uses `thrust::unary_function`, which CUDA 12.8's Thrust
  deprecates, and marian compiles with -Werror: add
  `-DCUDA_NVCC_FLAGS=-Xcompiler;-Wno-deprecated-declarations`. Upstream code,
  untouched by this port, and it fails identically without our changes.

The cuDNN half of the CUDA path could NOT be compile-checked: marian's cuDNN
code is written against the cuDNN 7 algorithm API, which cuDNN 8 removed, so
USE_CUDNN=ON does not build on the CUDA side with any cuDNN available today --
independent of this port. USE_CUDNN=OFF for the check.

### 2. Pooling backward staging buffer

Now `ensureBuffer(gradStaging_, gradStagingSize_, ...)` with the members beside
the existing pooling workspace, matching the convolution gradients. No
hipMalloc/hipFree per pooling node per batch.

### 3. MIOPEN_CALL

Prints `miopenGetErrorString(_s)`. Verified the decoder returns names
(`3 -> miopenStatusBadParm`, `8 -> miopenStatusUnsupportedOp`).

### 4. notes.md deferred section

Marked SUPERSEDED in place with a pointer to the 2026-08-10 sections; the
original text is kept for history.

### Re-test (gfx1100, GPU 0, USE_CUDNN=ON USE_NCCL=ON, incremental rebuild)

No regressions against the numbers before this round:

- operator_tests 603/603 assertions, 4 test cases (csr-dot and the fp16
  sections included) -- PASS.
- graph 10/10, attention 6/6, transformer 3/3, binary 9/9, fastopt 23/23,
  utils 8/8 -- PASS.
- rnn_tests 21/24 -- the three documented hipRAND-vs-cuRAND reference
  mismatches, unchanged.
- Convolution/pooling reference program (agent_space/conv_pool_check.cpp,
  relinked against the rebuilt library): ALL CHECKS PASSED -- conv forward,
  d/dinput, d/dkernel, d/dbias, max and average pooling forward+backward, and
  the character-encoder-shaped multi-channel conv.
- char-s2s training smoke, 1000 updates on GPU 0, exercising pooling backward
  with the cached buffer: cost 3.22 -> 1.70, gNorm finite throughout, no NaN.
  Matches the 1.70 of the previous round.

Multi-GPU RCCL and the end-to-end determinism gate were NOT re-run this round;
neither touches the three changed files, and both are the validator's to
confirm at this head.

### Rebuilding here: the submodules must be re-initialised

The sentencepiece deinit at the end of the last round leaves simd_utils and
simple-websocket-server missing too. Before any rebuild:
`git submodule update --init src/3rd_party/simd_utils
src/3rd_party/simple-websocket-server`, then the sentencepiece recovery
documented above (init fails on the unfetchable pin, then
`git -C src/3rd_party/sentencepiece checkout -f master`). NEVER stage the
resulting sentencepiece gitlink change.
