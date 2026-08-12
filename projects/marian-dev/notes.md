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
(`__CUDA_FP16_DISABLE_IMPLICIT_INTEGER_CONVERTS_FOR_HOST_COMPILERS__`).
[Corrected by the 2026-08-10 #2 review below: the converts became host-visible
in 12.2, not 12.8; the affected range is 11.0 through 12.1 plus any 12.2+ build
that defines the disable macro.]  This
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
- char-s2s training smoke, 1000 updates on GPU 0, exercising the convolution
  wrapper and training as a whole: cost 3.22 -> 1.70, gNorm finite throughout,
  no NaN.  Matches the 1.70 of the previous round.  It does NOT reach
  `PoolingWrapper`: the character encoder calls `pooling_with_masking`
  (`PoolingWithMaskingOp`, its own kernels, no MIOpen).  The evidence for the
  cached pooling buffer is the reference program above.

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

## Review 2026-08-10 #2 (reviewer, linux-gfx1100, fork moat-port 1381ed77..29ec0725)

Verdict: CHANGES REQUESTED. The three code fixes are correct -- I re-derived
the sparse-GEMM fault and the fix myself, and the pooling and MIOPEN_CALL
changes hold up. What blocks is the TEXT around them: the CUDA version
boundary is wrong in the skill lesson this branch publishes and in an upstream
commit message, and the pooling commit's Test Plan credits a program that does
not reach the changed code.

### 1. The "before 12.8" boundary is wrong -- it is 12.2 (blocker: it ships twice)

`.claude/skills/cuda-to-rocm/references/fault-classes.md:333,336`,
`.claude/skills/cuda-to-rocm/references/validation.md:11`, and the body of
fork commit 32d65345 ("broke the CUDA build on every toolkit older than 12.8",
"CUDA headers before 12.8 keep `__half`'s integer converts behind
`#if defined(__CUDACC__)`", "CUDA 12.8 made those converts visible").

`__half`'s integer converts became host-visible in CUDA **12.2**, not 12.8. I
bisected it with the real headers, compiling the pre-fix template shape
(`(ScalarType)0` with `ScalarType = __half`, host `g++ -std=gnu++17`, no
`__CUDACC__`) against each release's `cuda_fp16.h`/`.hpp`:

```
12.1 (/opt/conda/envs/cuda-12.1-hdr, CUDART_VERSION 12010)  -> ambiguous, bug present
12.2.53  (nvidia-cuda-runtime-cu12 wheel headers)           -> compiles
12.2.140 / 12.3.101 / 12.4.127                              -> compiles
12.8 (/opt/conda/envs/cuda-12.8, CUDART_VERSION 12080)      -> compiles
```

The mechanism is visible in the headers: 12.1 gates the integer constructors
on `#if defined(__CUDACC__)` (cuda_fp16.hpp:224), while 12.2 onward gates them
on `#if !(defined __CUDA_FP16_DISABLE_IMPLICIT_INTEGER_CONVERTS_FOR_HOST_COMPILERS__)
|| (defined __CUDACC__)` (cuda_fp16.hpp:139) -- host-visible unless a project
opts out with that macro.

So the true affected range for this file is CUDA 11.0 through 12.1 (the header
is selected at `prod_sparse.cpp:15`, `CUDA_VERSION >= 11000`), plus any 12.2+
build that defines the disable macro. That is still a real break and the fix is
still right; only the range is wrong.

This originated in my own 2026-08-10 review (notes.md:1121-1123), which
asserted the 12.8 boundary from a single 12.8-compiles observation; the porter
carried it into the commit body and then into the skill. Correct all four
places -- the two skill lines, the commit body of 32d65345, and leave this
section as the record for notes.md. The skill is the one that must not merge
wrong: it tells every future porter which toolkit to check against, and "get
anything older than 12.8" versus "get 12.1 or older" is the difference between
a check that reproduces and one that quietly passes.

### 2. The pooling Test Plan credits a program that does not run the changed code

Body of fork commit 4ff41294: "Character-CNN training, the only consumer of the
pooling path, converges as before over 1000 updates", and notes.md:1328
("char-s2s training smoke ... exercising pooling backward with the cached
buffer").

`PoolingWrapper` is reached only through `avg_pooling`/`max_pooling`
(`src/graph/expression_operators.cpp:966,977` -> `PoolingOp`,
`src/graph/node_operators_unary.h:1344`). The character encoder does not use
them: `src/layers/convolution.h:64` calls `pooling_with_masking`, which is
`PoolingWithMaskingOp` (`src/graph/node_operators_unary.h:1383`) with its own
`PoolingWithMaskingForward/Backward` kernels and no MIOpen wrapper at all. The
only in-tree caller of `max_pooling` is `src/examples/mnist/model_lenet.h:50`.
The char-CNN run does exercise `ConvolutionWrapper`, so it is evidence for the
port, but not for this commit.

The real evidence for this change is the standalone reference program (max and
average pooling forward and backward against hand-computed values), which the
same body already cites. Drop the char-CNN sentence from 4ff41294's Test Plan
or re-attribute it to the convolution path, and fix the notes line. An upstream
maintainer who knows this codebase will check what `pooling_with_masking` is.

### 3. The cached buffer does not survive a training step, so the comment overstates it

`src/tensors/gpu/cudnn_wrappers.cu:457-459` ("The buffer is a cached member so
that a training step does not allocate and free device memory per node") and
the matching paragraph in 4ff41294's body.

`PoolingWrapper` is a by-value member of `PoolingOp`
(`src/graph/node_operators_unary.h:1379`), and node objects do not outlive one
graph build: `ExpressionGraph::clear()` drops `nodesForward_`/`nodesBackward_`
and `topNodes_` (`src/graph/expression_graph.h:726-734`), and it runs per batch
via `EncoderDecoder::clear` (`src/models/encoder_decoder.cpp:180`, reached from
`build`/`stepAll` at :236). `backward` is called once per node per build, so a
buffer first allocated there is allocated exactly as often as the old local was
-- one `hipMalloc` per pooling node per batch either way, with the `hipFree`
merely deferred to node destruction.

The convolution case is different and the pattern is right there: one buffer
serves three gradients inside a single `backward` (`cudnn_wrappers.cu:290-294`,
used at :315, :330, :336), so caching genuinely removes two allocations per
call. Keep the pooling change (consistency with the convolution path is a fine
reason on its own), but say what is true: it is one allocation per wrapper
rather than per call, and the wrapper lives for one graph build. If the intent
really is to amortize across batches, the buffer has to hang off something with
a longer life than the node, which is a larger change than this commit.

I reasoned this from the code above rather than instrumenting a run; if a
counted `hipMalloc` trace over two batches shows otherwise, say so and this
one drops.

### Verified against the code (the fixes themselves are sound)

- `prod_sparse_cu11.h:188,256` are both `if constexpr` and :191 compares
  `(ScalarType)0.f`. `resultNeedsCast()` is `static constexpr`
  (:34, :39), the function is a member of a class template, so the discarded
  arm is not instantiated -- which is what unbreaks the CUDA path. I
  reproduced both directions with the real 12.1 headers: the pre-fix form
  fails with "call of overloaded `__half(int)` is ambiguous", the post-fix form
  compiles clean.
- HIP semantics unchanged by the conversion: `resultNeedsCast()` is a compile
  time constant on both paths, so `if constexpr` selects exactly the branch the
  runtime `if` took -- true for half on ROCm, false for float on ROCm and for
  everything on CUDA.
- No other int-to-`ScalarType` conversion of this class remains, in that file or
  in any file the branch touches (`(ScalarType|ElementType|half|__half)` cast of
  an integer literal: none). `ScalarType alpha = 1.0` at :181 is a double, an
  exact match for `__half(double)`, which 12.1 does expose to the host --
  compile-checked.
- Pooling buffer: `ensureBuffer` only grows (`cudnn_wrappers.cu:57-62`) and is
  called with the current `xGrad->shape().elements() * sizeof(float)` on every
  call (:460-462), so a growing shape reallocates and a shrinking one reuses a
  larger buffer, which is safe because `miopenPoolingBackward` runs with beta=0
  over `xDesc` only. No aliasing with the convolution staging: `gradStaging_` is
  a separate member of a separate class (`cudnn_wrappers.h:114` in
  `ConvolutionWrapper`, :154 in `PoolingWrapper`; neither derives from the
  other), each wrapper is owned by value by its own node, and within
  `PoolingWrapper` the staging is distinct from `workspace_`, which pooling
  backward passes to MIOpen in the same call (:477) -- the two are never the
  same allocation.
- `MIOPEN_CALL` (`cudnn_wrappers.cu:29-38`) now prints
  `miopenGetErrorString(_s)` (declared `/opt/rocm/include/miopen/miopen.h:138`)
  and is otherwise byte-for-byte the same control flow: same `do/while(0)`,
  same print-and-continue with no abort and no return, and the same message
  format as the cuDNN macro it mirrors (:509-517).
- notes.md:114-121 marks the old deferred section SUPERSEDED with a pointer to
  the current sections and keeps the history. The four findings of the previous
  review are answered point by point at notes.md:1248-1312.
- The lesson's C++ claims are right where the version is wrong: a plain `if` in
  a template does instantiate both arms, and `if constexpr` does not instantiate
  the discarded arm once the condition is no longer value-dependent. Filing is
  right too -- the fault class in fault-classes.md, the "check against an old
  toolkit" habit in validation.md under the nvcc CUDA-build gate, cross-linked.
- Hygiene: `jargon.py --port marian-dev` clean over the whole branch; the three
  new titles are 48, 48 and 52 chars, all `[ROCm]`; Claude named in each body;
  no `Co-Authored-By` and no noreply trailer; ASCII throughout the messages and
  the diff; no internal account, host or path (the toolkit paths are
  placeholders); fork worktree clean.
- Fault classes: no device code in these three commits -- no kernel, no warp
  intrinsic, no literal 32, no texture. The C++17 `if constexpr` is available on
  every compiler this build uses (`CMakeLists.txt:15`, `:335`, `:483`, and
  `:684` for nvcc), and `prod_sparse_cu11.h` is included only by the host TU
  `prod_sparse.cpp:16`.

### Taken on trust

The 161-target nvcc 12.8 configure-build-link (the `build-cuda` tree is gone
from this checkout, so there is nothing left to inspect) and the operator/graph
suite numbers from the porter's re-run. Both are the validator's to confirm at
this head; their absence here is not part of this verdict. The two conda envs
the notes name do exist and their versions match, and the fault and its fix I
reproduced independently, so the part of the claim that carries the fix is not
on trust.

## Porter response to review 2026-08-10 #2 (linux-gfx1100)

All three findings accepted; all three were text, and the compiled code is
untouched apart from one comment.  New fork head `4d25e3f1`
(1381ed77 -> 82755003 -> d4bc6433 -> 4d25e3f1), replacing
32d65345/4ff41294/29ec0725.  `git diff 29ec0725 4d25e3f1` is 4 insertions and 2
deletions in `src/tensors/gpu/cudnn_wrappers.cu`, all of them inside one comment
block; every other tree object is identical.  ba0ec806, the `validated_sha` of
all three completed archs, is still a reachable ancestor, and no arch had
validated any of the three rewritten commits.

**The delta is comment-only and message-only.**  `moatlib classify marian-dev
29ec0725 4d25e3f1` reports `class=comment-only ... inert=True`, so the
regression guard treats it as behavior-preserving.  It does not change what the
completed archs are owed: their `validated_sha` is still ba0ec806, which was
already behind before this round, so gfx90a, gfx1100 and windows-gfx1201 read
`revalidate` for the functional commits above ba0ec806 exactly as they did
yesterday.  A validator confirming those may treat the 29ec0725 -> 4d25e3f1 step
itself as free, and `utils/codeobj_diff.py` against 29ec0725 will show binary
equivalence if anyone wants it proven rather than classified.  No rebuild and no
GPU re-run was done for this round, deliberately: nothing compiled changed.

1. **CUDA boundary is 12.2, not 12.8.**  Corrected in the skill
   (`references/fault-classes.md`, now naming the 12.2 gate change to
   `!defined(__CUDA_FP16_DISABLE_IMPLICIT_INTEGER_CONVERTS_FOR_HOST_COMPILERS__)
   || defined(__CUDACC__)`, the 11.0-12.1 affected range plus any 12.2+ build
   defining the disable macro, and "compile-check against 12.1 headers or
   older"), in `references/validation.md:11`, in the body of 82755003 (was
   32d65345), and at notes.md:1121 where the wrong boundary was first written.
   The reviewer's bisection is the record; I did not re-run it.
2. **Pooling Test Plan.**  d4bc6433's Test Plan now credits the standalone
   reference program as the evidence for the changed code, states that
   avg_pooling/max_pooling are this wrapper's only callers with the MNIST LeNet
   example as the only in-tree user, and re-attributes the char-s2s run to the
   convolution wrapper and to training as a whole, saying explicitly that its
   pooling goes through `pooling_with_masking`.  notes.md:1328 fixed the same
   way.
3. **Cache comment.**  `cudnn_wrappers.cu:457` now says the member is for
   consistency with the convolution path and that here it saves only the free,
   because the wrapper lives on its node and the graph drops nodes for the next
   batch, so the allocation still happens once per pooling node per batch.
   d4bc6433's body carries the same correction and notes that amortizing across
   batches needs a buffer that outlives a node.  The code is unchanged, as the
   review allowed.

`python3 utils/jargon.py --port marian-dev`: clean over the whole branch.

## Review 2026-08-10 #3 (reviewer, linux-gfx1100, fork moat-port 29ec0725 -> 4d25e3f1)

Verdict: CHANGES REQUESTED, narrowly.  Scope of this round was the three text
corrections from review #2 and nothing broader.  Two of the three landed and are
correct; the third (the pooling cache comment) replaced an overstatement with a
smaller one that the surrounding code still contradicts.  One further wording
point in the skill entry that merges with this branch.  Both are wording; no
code change is being asked for, so the delta stays comment/message-only and
`classify` should still read inert.

### 1. "It saves only the free" -- the free is not saved, it is deferred

`src/tensors/gpu/cudnn_wrappers.cu:457-461` ("the buffer is a cached member for
consistency with that path, but here it saves only the free") and the matching
sentence in d4bc6433's body ("What this removes is the free rather than the
allocation").

`DeviceBuffer` is `std::shared_ptr<void>` with a `hipFree` deleter
(`cudnn_wrappers.h:37`, `cudnn_wrappers.cu:50-53`), so destroying the wrapper
frees the buffer.  `PoolingWrapper` is a by-value member of `PoolingOp`
(`node_operators_unary.h:1379`), every node is held by `nodesForward_`
(`expression_graph.cpp:51`) until `ExpressionGraph::clear()` drops it
(`expression_graph.h:726-734`), and that runs per batch.  So the free happens
once per pooling node per batch either way: with the old local it ran at the end
of `backward`, with the cached member it runs at node teardown.  The counts of
`hipMalloc` and `hipFree` per batch are unchanged by this commit; only the
moment of the free moves.  The comment's own reasoning says as much -- the node
being dropped for the next batch is exactly what performs the free -- so a
maintainer reading the two clauses together finds them in tension.

Fix both places to say what happens: the allocation still occurs once per
pooling node per batch, and the free moves out of the backward call to node
teardown.  The consistency-with-the-convolution-path rationale carries the
change on its own and is already there.

### 2. The skill entry's "11.0" floor is marian's, not the fault's

`.claude/skills/cuda-to-rocm/references/fault-classes.md:336-338` ("so the
affected range is 11.0 through 12.1 plus any 12.2-or-newer build that defines
the disable macro").

11.0 is where *this project* starts using the header that carries the cast
(`prod_sparse.cpp:15`, `CUDA_VERSION >= 11000`; `prod_sparse_cu10.h` has no
`(ScalarType)0`), so it is right in 82755003's body, which is about marian.  The
skill entry states it as the general range of the fault class, and nothing in
this branch establishes that releases older than 11.0 are unaffected -- they
were not tested in either direction, and the sentence just above it already says
the correct general thing ("CUDA headers up to 12.1 keep `__half`'s integer
converts behind `#if defined(__CUDACC__)`").  Say "every release up to 12.1"
there, or name 11.0 as marian's own floor.  The actionable instruction that
follows ("compile-check against 12.1 headers or older") is right as it stands.

### Verified this round

- CUDA boundary: correct as now written everywhere it appears.  I re-read the
  real headers rather than taking the bisection on trust:
  `/opt/conda/envs/cuda-12.1-hdr/include/cuda_fp16.hpp:224` gates the `__half`
  integer constructors (`short`, `unsigned short`, `int`, `unsigned int`,
  `long long`, `unsigned long long`) on `#if defined(__CUDACC__)` alone, and
  `/opt/conda/envs/cuda-12.8/targets/x86_64-linux/include/cuda_fp16.hpp:141`
  gates the same block on `#if !(defined
  __CUDA_FP16_DISABLE_IMPLICIT_INTEGER_CONVERTS_FOR_HOST_COMPILERS__) ||
  (defined __CUDACC__)`.  82755003's body, `fault-classes.md:333-341`,
  `validation.md:11` and the bracketed correction at notes.md:1124-1126 all
  state 12.2, the opt-out macro and the 11.0-12.1 range consistently, and the
  macro spelling matches the header.  The 11.0 floor is right for marian (see
  finding 2 for the skill).
- Pooling Test Plan (d4bc6433): the standalone reference program is now the
  credited evidence; "avg_pooling and max_pooling are the only callers of this
  wrapper" holds (`PoolingOp` is constructed only there), and "in this tree only
  the MNIST LeNet example uses them" holds -- `model_lenet.h:50` is the only
  live caller, `tests/pooling.cpp` has its pooling entirely commented out, and
  `tests/conv_char.cu:69` uses a `MaxPooling` class that has no definition
  anywhere in the tree and is not in any CMake target list.  char-s2s is
  re-attributed to the convolution wrapper and training as a whole, with
  `pooling_with_masking` named as where its pooling goes; no claim that
  char-CNN reaches this file remains.  notes.md:1330-1335 matches.
- Tree identity: `git diff 29ec0725 4d25e3f1 --stat` is one file,
  `src/tensors/gpu/cudnn_wrappers.cu`, 4 insertions and 2 deletions, all inside
  the one comment block.  ba0ec806 and 1381ed77 are both reachable ancestors of
  4d25e3f1.
- Hygiene: the three titles are byte-identical to the ones they replaced (48,
  48, 52 chars, all `[ROCm]`); Claude named in each body; no `Co-Authored-By`
  and no noreply trailer; ASCII throughout the three messages; no in-house
  vocabulary in the new body text (`jargon.py --port marian-dev` clean over the
  whole branch); no internal account, host or path -- the toolkit paths in
  82755003's Test Plan are placeholders; fork worktree clean at 4d25e3f1.
- Skill edit is surgical: 7d49e68 touches only the two boundary sentences in
  `fault-classes.md` and the one in `validation.md`.  The `if constexpr` claims,
  the float-literal advice, the cross-link and the surrounding entries are
  unchanged, and the 12.1-header conda recipe in `validation.md` still matches
  the env that exists.
- Not re-reviewed, deliberately: the compiled code, which is unchanged from
  29ec0725 apart from the one comment, and whose review is in #2.  No GPU run is
  expected for a comment-only delta and its absence is not part of this verdict.

## Porter response to review 2026-08-10 #3 (linux-gfx1100)

Both findings were wording; no compiled code changed.  The top two fork commits
were rewritten in place (`git reset --hard 82755003`, re-apply the pooling
commit with the corrected comment and body, cherry-pick the MIOpen-names commit
unchanged), so the branch is now
`82755003 -> fd8ef1ff -> 1d0822bd` where it was `82755003 -> d4bc6433 ->
4d25e3f1`.  `ba0ec806` and `1381ed77` remain reachable ancestors, and the two
titles are byte-identical to the ones they replace.

### 1. The free is deferred, not saved -- fixed in both places

Agreed, and the correction is the reviewer's own reading: `DeviceBuffer` is a
`shared_ptr<void>` with a `hipFree` deleter, the wrapper is a by-value member of
`PoolingOp`, and `ExpressionGraph::clear()` drops the node per batch, so the
malloc/free counts per batch are unchanged by the commit and only the moment of
the free moves.  `cudnn_wrappers.cu:457-462` now reads:

```
  // As with the convolution gradients, the accumulating form is a pooling
  // backward into staging plus a tensor add. The buffer is a cached member for
  // consistency with that path, but the wrapper lives on its node and the
  // graph drops it for the next batch, so per batch nothing is saved: the
  // allocation still happens once per pooling node, and the free moves from
  // the end of backward to node teardown.
```

The commit body's second paragraph lost the "What this removes is the free
rather than the allocation" claim and now says the same thing the comment does
-- per batch this saves nothing, the allocation still happens once per pooling
node per batch, the free moves from the end of the backward call to node
teardown -- and states the consistency-with-the-convolution-gradient-path
rationale as what carries the change.  Nothing else in that message moved; the
Test Plan and the title are untouched.

### 2. The skill's 11.0 floor was marian's -- reworded

Agreed.  Nothing on this branch tested toolkits older than 11.0 in either
direction, and 11.0 is only where `prod_sparse.cpp:15` starts selecting the
header that carries the cast.  `fault-classes.md:337` now says "so every release
up to 12.1 is affected, plus any 12.2-or-newer build that defines the disable
macro", dropping the floor rather than restating it as marian's -- the sentence
above it already scopes the fault correctly, and the bisected upper boundary is
the actionable half.  82755003's body keeps 11.0, where it is right, because
that message is about marian.

### Verification (no rebuild, no GPU run)

The delta is comment-and-message-only, so nothing was rebuilt and no GPU test
was run; the compiled evidence is the one recorded for `ba0ec806` and the
per-arch validation states are unchanged by this round.

- `git diff 4d25e3f1 1d0822bd --stat` is one file,
  `src/tensors/gpu/cudnn_wrappers.cu`, 4 insertions and 3 deletions, all inside
  the one comment block.
- `changeclass.py projects/marian-dev/src 4d25e3f1 1d0822bd` reads
  `class=comment-only arch_independent=False inert=True` -- inert, but flagged
  for the line-shift hazard, since `MIOPEN_CALL` prints `__LINE__` and the
  comment grew by a line.  So no arch is carried forward on the source class
  alone; that costs nothing here, because every arch's `validated_sha` was
  already `ba0ec806` and behind the head before this round.
  `windows-gfx1101`'s `failed_sha` stays `ba0ec806`, so its failure is retired
  by the head move and the arch is owed a validation rather than another fix.
- `python3 utils/jargon.py --port marian-dev` clean over the whole branch.
- Both rewritten titles byte-identical to their predecessors (48 and 52 chars);
  Claude named in each body; no `Co-Authored-By` and no noreply trailer; ASCII
  throughout.
- Fork worktree clean at `1d0822bd`; pushed with `--force-with-lease`.

## Review 2026-08-10 #4 (reviewer, linux-gfx1100, fork moat-port 4d25e3f1 -> 1d0822bd)

Verdict: PASS.  No findings.  Scope was the two wording corrections from review
#3 and the standing invariants; rounds 1-3 covered the compiled code, which is
byte-identical to what they reviewed apart from one comment block.

### 1. The pooling cache comment is now exactly true

`src/tensors/gpu/cudnn_wrappers.cu:457-462` and fd8ef1ff's second paragraph both
now say: per batch nothing is saved, the allocation still happens once per
pooling node, and the free moves from the end of `backward` to node teardown.
That matches the ownership chain end to end, checked against the code rather
than against the porter's account of it:

- `DeviceBuffer` is `std::shared_ptr<void>` (`cudnn_wrappers.h:37`) and
  `allocateDeviceBuffer` (`cudnn_wrappers.cu:40-52`) attaches a `hipFree`
  deleter, so the last owner going away IS the free.
- `PoolingWrapper pooling_` is a by-value member of `PoolingOp`
  (`node_operators_unary.h:1379`), so the wrapper and its `gradStaging_` die
  with the node.
- Nodes are held by `nodesForward_` (`expression_graph.cpp:51`) and dropped by
  `ExpressionGraph::clear()` (`expression_graph.h:726-734`), which runs per
  batch on both live paths -- `encoder_decoder.cpp:236` for training and
  `model_lenet.h:15,24` for the MNIST LeNet example, the only live caller of
  avg_pooling/max_pooling.  `PoolingOp` is not memoized (`memoize_` defaults
  false, `node.h:26`), so nothing keeps the node past a clear.
- A fresh node starts with `gradStagingSize_ == 0`, so `ensureBuffer`
  (`cudnn_wrappers.cu:57-61`) allocates on the first `backward`, and `backward`
  runs once per node per backprop.  Against `82755003:cudnn_wrappers.cu:459-460`
  (a local `DeviceBuffer` freed at scope exit) the malloc and free counts per
  batch are identical; only the moment of the free moves.

Neither over- nor understated: it does not claim a saving, and it does not
claim the free was eliminated.  The consistency-with-the-convolution-path
rationale carries the change, and both the comment and the body say so.

### 2. The skill's 11.0 floor is gone and nothing adjacent moved

`fault-classes.md:337` now reads "so every release up to 12.1 is affected, plus
any 12.2-or-newer build that defines the disable macro".  0f2eec4 is a
one-line, one-file change; the `if constexpr` explanation, the `(ScalarType)0`
trap, the `#if defined(__CUDACC__)` boundary sentence above it, the macro
spelling, the float-literal advice and the "compile-check against 12.1 headers
or older ... 12.1 fails and 12.2.53 onward compiles" instruction are all
unchanged.  82755003's body keeps "11.0 through 12.1", which is right there:
`prod_sparse.cpp:15` gates the header carrying the cast on
`CUDA_VERSION >= 11000`, so that floor is marian's own and the message is about
marian.

### Invariants

- `git diff 4d25e3f1 1d0822bd --stat`: one file,
  `src/tensors/gpu/cudnn_wrappers.cu`, 4 insertions 3 deletions, all inside the
  one comment block.  `changeclass.py` reads
  `class=comment-only arch_independent=False inert=True`.
- `ba0ec806`, `1381ed77` and `82755003` are all reachable ancestors of
  `1d0822bd` (`git merge-base --is-ancestor`).
- Titles byte-identical to the ones they replaced ("[ROCm] Cache the pooling
  backward staging buffer", 48; "[ROCm] Report MIOpen failures by name, not by
  number", 52); longest title on the branch is 61 chars, every one `[ROCm]`.
- `1d0822bd`'s body diffs empty against `4d25e3f1`'s.
- All 14 branch commits name Claude; no `Co-Authored-By` and no noreply
  trailer; ASCII throughout; no internal account, host or path in messages or
  in the branch diff (the toolkit paths in 82755003's Test Plan are
  placeholders).
- `python3 utils/jargon.py --port marian-dev` clean over the whole branch.
- Fork worktree clean; `origin/moat-port` == local HEAD == `1d0822bd`.
- No GPU run this round and none expected: the delta cannot change compiled
  behavior beyond `__LINE__` in a `MIOPEN_CALL` diagnostic.  Its absence is not
  part of this verdict.

## Validation 2026-08-10 (linux-gfx1100, 4x AMD Radeon Pro W7800 48GB, real GPU)

Verdict: PASS. `linux-gfx1100` revalidated at head `1d0822bd` (was `ba0ec806`).
Host: 4x gfx1100 (RDNA3, wave32), ROCm 7.2.3. Fork clone verified clean and at
`origin/moat-port` == `1d0822bd` before and after the run (sentencepiece
gitlink recovered per notes above, never staged, deinited when done).

### Build

Submodules: `git submodule update --init src/3rd_party/simd_utils
src/3rd_party/simple-websocket-server`, then the sentencepiece recovery (init
fails on the unfetchable pin, `git -C src/3rd_party/sentencepiece checkout -f
master`, 1ca221c).

```
cmake -S projects/marian-dev/src -B agent_space/marian-build-gfx1100-full -G Ninja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCOMPILE_CUDA=ON -DUSE_CUDNN=ON -DUSE_NCCL=ON \
  -DUSE_FBGEMM=OFF -DCOMPILE_CPU=ON -DCMAKE_BUILD_TYPE=Release \
  -DCOMPILE_TESTS=ON -DUSE_MKL=OFF -DUSE_TCMALLOC=OFF -DUSE_DOXYGEN=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DBUILD_ARCH=native
cmake --build agent_space/marian-build-gfx1100-full -j64
```

294/294 targets, no errors (compile 3.9s configure + 108.2s build). All three
maintainer-requested features on: USE_CUDNN=ON (MIOpen), USE_NCCL=ON (RCCL).

### Unit suites (HIP_VISIBLE_DEVICES=0)

Matches the porter's numbers exactly:

- operator_tests: 603/603 assertions, 4 test cases -- PASS (csr-dot + fp16
  sections included).
- graph 10/10, attention 6/6, transformer 3/3, binary 9/9, fastopt 23/23,
  utils 8/8 -- PASS.
- rnn_tests 21/24 -- the three documented hipRAND-vs-cuRAND reference
  mismatches; not a regression.

### Conv/pool reference program

`agent_space/conv_pool_check.cpp` (the validation aid at
`/var/lib/jenkins/moat/agent_space/conv_pool_check.cpp`, not committed to the
fork) recompiled against this build's `libmarian.a`/`libmarian_cuda.a` with
the same flags/libs `test_prod` links with, and run on GPU 0:

```
conv forward             OK
conv grad wrt input      OK
conv grad wrt kernel     OK
conv grad wrt bias       OK
max pool forward         OK
max pool grad            OK
avg pool forward         OK
avg pool grad            OK
charcnn conv forward     OK
charcnn conv grads       OK (finite)
ALL CHECKS PASSED
```

### char-s2s training smoke (GPU 0)

`marian --type char-s2s --char-highway 0 --dim-emb 64 --dim-rnn 64
--enc-type bidirectional --enc-depth 1 --dec-depth 1 --mini-batch 16
--workspace 2048 --after 1000u --learn-rate 0.0001 --optimizer adam
--clip-norm 1 --seed 2222 --devices 0` against the same toy reverse-copy
corpus/vocab as the e2e gate (`/var/lib/jenkins/moat/agent_space/marian-e2e`).
Cost 3.22 (Up.100) -> 1.66 (Up.1000), gNorm finite throughout, no NaN/Inf in
the run -- matches the documented ~3.2 -> ~1.7 pattern (small run-to-run
numeric drift is expected; hipRAND streams and reduction order are not
required to be bit-identical across processes). Note: a first attempt through
`timeit.sh` was killed by the harness's default 2-minute foreground timeout
before the training's own ~142s completed; that killed attempt is NOT the
result reported here (GPU was confirmed idle both before and after, and the
attempt was discarded). The reported run was launched standalone, monitored
to completion, and is the sole writer of its output path.

### Multi-GPU RCCL (4x gfx1100, exclusive use of the host's four GPUs)

Per the skill's validation.md RCCL section: no other GPU job ran during this
window (`rocm-smi --showuse` confirmed all four idle immediately before
start).

```
marian --type transformer -t train.src train.tgt -m rccl_sync_validate.npz \
  --vocabs vocab.src.yml vocab.tgt.yml --dim-emb 64 --transformer-dim-ffn 128 \
  --transformer-heads 2 --enc-depth 2 --dec-depth 2 --after 3000u \
  --devices 0 1 2 3 --sync-sgd --overwrite
```

Log confirms `[comm] Using NCCL 4.7.7 for GPU communication` and
`[comm] NCCLCommunicators constructed successfully`. Synchronous SGD across
all 4 devices converges: cost 0.00568 @3000u (started from the same corpus as
the char-s2s/e2e gates). Decoding the resulting model on GPU 0 (beam=6)
reproduces the reverse-copy task exactly -- every output line is the exact
reverse of the corresponding input line in `test.src`. Not just a
start-and-not-crash: the communicators construct, training converges across
every device, and the trained model decodes correctly.

### Single-GPU e2e determinism gate (GPU 0)

Trained a fresh transformer (same toy task, `--after 3000u --devices 0`,
default async) to cost 0.0106, then decoded beam=6 twice on GPU and once on
CPU (`--cpu-threads 1`):

```
diff gpu1.out gpu2.out   # IDENTICAL
diff gpu1.out cpu.out    # IDENTICAL
```

GPU run1 == GPU run2 (deterministic) and GPU == CPU. The wave32 topk/nth_element
fix still holds at this head.

### CUDA no-regression gate: not re-run, and why that is sound

The porter's nvcc 12.8 compile-and-link check (161/161 targets, USE_CUDNN=OFF
because cuDNN 8 removed the algorithm API this file targets) was recorded at
`82755003`, two commits behind this head (`1d0822bd`). Checked whether that
gap matters rather than assuming it does not:
`git diff 82755003 1d0822bd --stat` touches exactly two files,
`src/tensors/gpu/cudnn_wrappers.cu` (34 lines) and
`src/tensors/gpu/cudnn_wrappers.h` (2 lines). Both hunks fall entirely inside
`#if defined(CUDNN) && defined(USE_HIP)` (the `.cu`, lines 7-508) or
`#if defined(USE_HIP)` (the `.h`, the `PoolingWrapper` HIP branch, lines
19-156) -- neither compiles when `USE_HIP` is not defined, i.e. never on the
CUDA path regardless of `USE_CUDNN`. So the object code nvcc would produce at
`1d0822bd` is identical to what it produced at `82755003`, and the recorded
161/161 link result still stands. No rebuild performed; this is a source-level
proof of the CUDA gate, not a skip on trust.

### Jargon and docs (finished-port gate)

- `python3 utils/jargon.py --port marian-dev`: clean.
- ROCm build documented in marian's own house style: `CHANGELOG.md` names the
  `-DUSE_HIP=on`/`-DCMAKE_HIP_ARCHITECTURES` switches and the hipBLAS,
  hipBLASLt, hipSPARSE, hipRAND, rocThrust, MIOpen (`-DUSE_CUDNN=on`), RCCL
  (`-DUSE_NCCL=on`) library set; `README.md`'s feature line now says ROCm/AMD
  alongside CUDA/NVIDIA. Full build instructions live off-repo (the project
  website) for CUDA too, so nothing is duplicated -- matches house style.

### Fork clone hygiene

`git status --porcelain` in `projects/marian-dev/src` clean before and after
(the sentencepiece gitlink recovery is transient and deinited at the end,
never staged). `origin/moat-port` == local HEAD == `1d0822bd` throughout.

### Verdict

`linux-gfx1100`: **completed** at `validated_sha` = `1d0822bd`. All 5
requested checks pass; no regressions against the porter's numbers; RCCL and
MIOpen both exercised for real on GPU, not compile-only.

## Validation 2026-08-11 (linux-gfx90a, 4x AMD Instinct MI250X, real GPU)

Verdict: PASS. `linux-gfx90a` revalidated at head `1d0822bd` (was `ba0ec806`,
carried forward from a cosmetic pin-removal; this run is a full real-GPU
re-run, not a carry-forward, per the eight functional commits since then).
Host: 4x gfx90a (MI250X, CDNA2, wave64), ROCm 7.2.1. No fork clone existed for
this project on this host; cloned `AMD-Ecosystem/marian-dev` fresh, checked
out `moat-port`, confirmed HEAD == `1d0822bd` before and after the run.

### Build

Submodules: `git submodule update --init src/3rd_party/simd_utils
src/3rd_party/simple-websocket-server`, then sentencepiece. The direct
`git submodule update --init src/3rd_party/sentencepiece` hung for several
minutes on this host (slow path to GitHub, not the documented unfetchable-pin
error) -- worked around by cloning `marian-nmt/sentencepiece` to a scratch dir
first (`/tmp/sp_test`) and re-running the submodule update with
`--reference /tmp/sp_test`, which reuses those objects locally. It still hits
the documented `fatal: remote error: upload-pack: not our ref
f006008f97c8a724d2dee306fb5347b109dbb893` for the pinned commit, landing on
`master` (1ca221c, v0.1.95) exactly as notes above describe. One new wrinkle:
after `git submodule update --init --reference`, the submodule's `git log`
was fine but `git status` showed every tracked file as staged-deleted (empty
working tree) -- `git -C src/3rd_party/sentencepiece reset --hard HEAD`
restored the files; a bare `checkout -- .` fails with "pathspec '.' did not
match" in that half-initialized state, `reset --hard` is what actually works.

```
cmake -S projects/marian-dev/src -B agent_space/marian-build-gfx90a-full -G Ninja \
  -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCOMPILE_CUDA=ON -DUSE_CUDNN=ON -DUSE_NCCL=ON \
  -DUSE_FBGEMM=OFF -DCOMPILE_CPU=ON -DCMAKE_BUILD_TYPE=Release \
  -DCOMPILE_TESTS=ON -DUSE_MKL=OFF -DUSE_TCMALLOC=OFF -DUSE_DOXYGEN=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DBUILD_ARCH=native
cmake --build agent_space/marian-build-gfx90a-full -j64
```

294/294 targets, no errors (configure 5.2s, build 107.7s -- much of the
object cache was warm from ccache-equivalent reuse across earlier attempts on
this host). All three maintainer-requested features on: USE_CUDNN=ON
(MIOpen), USE_NCCL=ON (RCCL). `git status --porcelain` in the fork clone:
only the sentencepiece gitlink shown modified (never staged), clean
otherwise, both before and after the full run.

### Unit suites (HIP_VISIBLE_DEVICES=0)

Matches the gfx1100 numbers exactly:

- operator_tests: 603/603 assertions, 4 test cases -- PASS (csr-dot + fp16
  affine sections included; these are the two surfaces the eight-commit delta
  fixed, and both are exercised and green here for the first time on gfx90a).
- graph 10/10, attention 6/6, transformer 3/3, binary 9/9, fastopt 23/23,
  utils 8/8 -- PASS.
- rnn_tests 21/24 -- the three documented hipRAND-vs-cuRAND reference
  mismatches (rnn_tests.cpp:93, "Simple RNN"), byte-identical to every prior
  platform's failure signature. Not a regression.
- App tests: test_logger, test_dropout, test_prod, test_pooling all exit 0.

### Conv/pool reference program (MIOpen correctness, not just graph construction)

`src/tests/pooling.cpp` (the CUDNN app test) only builds a graph and never
calls forward/backward -- it proves nothing about MIOpen numerics. Wrote a
standalone check (`agent_space/conv_pool_check.cpp`, not committed to the
fork; a validation aid) that builds tiny hand-computable `ConvolutionOp` /
`PoolingOp` cases directly and checks forward AND backward against values
worked out by hand, compiled and linked against this build's
`libmarian.a`/`libmarian_cuda.a` with the same flags `test_prod` links with
(`-no-pie` needed; `libmarian_cuda.a` has PIC-incompatible relocations
otherwise).

```
conv forward             OK   (x*3+1 elementwise on a 2x2 map)
conv grad wrt input      OK   (= kernel weight, broadcast)
conv grad wrt kernel     OK   (= sum(x))
conv grad wrt bias       OK   (= output element count)
max pool forward         OK   (1x2 non-overlapping windows)
max pool grad            OK   (routes to the argmax element only)
avg pool forward         OK   (1x2 non-overlapping windows)
avg pool grad            OK   (spreads 1/window_size evenly)
ALL CHECKS PASSED
```

**Bug found and worked around, not a ROCm regression:** the first version of
this check built `max_pooling(x,...)` and `avg_pooling(x,...)` on the SAME
input `x` in the SAME graph, and `avg pool forward` came back identical to
`max pool forward` (got `5 4 8 7`, wanted `3 3 5.5 6.5`). Root cause is in
upstream marian, not the port: `ExpressionGraph::add()` runs short-term CSE
(`findOrRemember` in `expression_graph.h`) unconditionally, keyed on
`node->hash()`/`node->equal()`. `PoolingOp::type()`
(`node_operators_unary.h`) returns the literal string `"layer_pooling"` for
BOTH `"avg"` and `"max"` mode and does not override `equal()`/`hash()` to
account for the `mode` field, unlike e.g. `ScalarAddNodeOp` which does
override `equal()` for its own scalar member. So a second `PoolingOp` with an
identical child and the same (empty, auto-assigned) name looks `equal()` to
the first, and the CSE silently aliases the two nodes -- `avg_pooling`
returns `max_pooling`'s result. This is `#ifdef CUDNN`-gated code, identical
on CUDA, and triggers only when both pooling modes are applied to the same
input in the same graph: confirmed dead in practice, `avg_pooling` has zero
callers anywhere in the tree and `max_pooling` has exactly one
(`examples/mnist/model_lenet.h`), so no shipped model path ever does this.
Not filed as a bug report (out of scope for this port, pre-existing on CUDA
too, no shipped path affected) -- worked around in the check by giving each
pooling mode its own graph, which is what isolates the numbers above as a
real MIOpen correctness result rather than an artifact of the CSE bug.

### Single-GPU e2e determinism gate (GPU 0)

Toy reverse-copy corpus regenerated fresh (agent_space is gitignored,
per-host; `agent_space/marian-e2e-gfx90a/{train,test}.{src,tgt}`, 1000
train sentences over a 30-word vocab, vocab.yml auto-built by marian).
Trained a transformer to cost 0.00056 @3000u, decoded beam=6 twice on GPU and
once on CPU:

```
marian --type transformer -t train.src train.tgt -m model.npz \
  --vocabs vocab.src.yml vocab.tgt.yml --dim-emb 64 --transformer-dim-ffn 128 \
  --transformer-heads 2 --enc-depth 2 --dec-depth 2 --after 3000u --devices 0 --seed 1234
marian-decoder -m model.npz -v vocab.src.yml vocab.tgt.yml -i test.src -b 6 --devices 0 > gpu1.out
marian-decoder -m model.npz -v vocab.src.yml vocab.tgt.yml -i test.src -b 6 --devices 0 > gpu2.out
marian-decoder -m model.npz -v vocab.src.yml vocab.tgt.yml -i test.src -b 6 --cpu-threads 1 > cpu.out
diff gpu1.out gpu2.out   # IDENTICAL
diff gpu1.out cpu.out    # IDENTICAL
```

GPU run1 == GPU run2 (deterministic) and GPU == CPU. Every one of the 10 test
lines decodes to the exact word-reversal of its input. The wave64
topk/nth_element fix still holds at this head.

### Multi-GPU RCCL (4x gfx90a, exclusive use of the host's four GPUs)

`rocm-smi --showuse` confirmed all four GPUs idle (0% activity) immediately
before start; no other GPU job ran during the window.

```
marian --type transformer -t train.src train.tgt -m rccl_sync_validate.npz \
  --vocabs vocab_rccl.src.yml vocab_rccl.tgt.yml --dim-emb 64 --transformer-dim-ffn 128 \
  --transformer-heads 2 --enc-depth 2 --dec-depth 2 --after 3000u \
  --devices 0 1 2 3 --sync-sgd --seed 1234 --overwrite
```

Log confirms `[comm] Using NCCL 4.7.7 for GPU communication` and `[comm]
NCCLCommunicators constructed successfully`; `[training] Batches are
processed as 1 process(es) x 4 devices/process`. Synchronous SGD across all 4
devices converges (cost 0.00039 @3000u). Decoding the trained model on GPU 0
(beam=6): 9/10 test lines are the exact reversal of the input; one line
(input starts with the repeated token `w8 w8`) is missing one trailing `w8`
in the output. Confirmed this is a toy-model generalization artifact, not a
GPU/RCCL correctness bug: decoding the SAME model on CPU (`--cpu-threads 1`)
produces a byte-identical output file to the GPU decode, including the same
near-miss on that one line -- GPU and CPU agree exactly on what the model
(under-trained on repeated-token sequences, at only 3000 updates on a 1000-
sentence toy corpus) actually computes. Not just start-and-not-crash: the
communicators construct, training converges across every device, and
GPU-vs-CPU decode of the resulting model is bit-identical.

### char-s2s training smoke (GPU 0, exercises MIOpen convolution via CharConvPooling)

`marian --type char-s2s --char-highway 0 --dim-emb 64 --dim-rnn 64
--enc-type bidirectional --enc-depth 1 --dec-depth 1 --mini-batch 16
--workspace 2048 --after 1000u --learn-rate 0.0001 --optimizer adam
--clip-norm 1 --seed 2222 --devices 0` against the same toy reverse-copy
corpus. Cost 2.07 @Up.1000, gNorm 2.80 (finite), no NaN/Inf anywhere in the
log. Matches the documented convergence pattern from the gfx1100 run.

### CUDA no-regression gate: not re-run, already recorded at this head

The porter's nvcc 12.8 compile-and-link check (161/161 targets) was recorded
at `82755003`; the gfx1100 validator proved by source diff
(`git diff 82755003 1d0822bd --stat`) that the two files touched between
`82755003` and this head (`cudnn_wrappers.cu`/`.h`) fall entirely inside
`#if defined(USE_HIP)`/`#if defined(CUDNN) && defined(USE_HIP)` guards, so
the CUDA-path object code at `1d0822bd` is identical to `82755003`. That
proof is head-sha-scoped, not arch-scoped, so it stands unchanged here; no
second nvcc run performed.

### Jargon and docs (finished-port gate)

`python3 utils/jargon.py --port marian-dev`: clean. Docs already verified in
the gfx1100 validation at this same head (CHANGELOG.md / README.md); nothing
about this arch changes that.

### Fork clone hygiene

`git status --porcelain` in `projects/marian-dev/src`: only the
sentencepiece gitlink shown modified (documented, never staged), clean
otherwise, before and after the full run. `origin/moat-port` == local HEAD ==
`1d0822bd` throughout.

### Verdict

`linux-gfx90a`: **completed** at `validated_sha` = `1d0822bd`. All five
newly-enabled surfaces (hipSPARSE csr-dot, fp16 affine scaling, MIOpen
conv/pool, RCCL multi-GPU, and the pre-existing wave64 topk fix) verified for
real on this host's 4x MI250X; no regressions against the gfx1100 numbers at
the same head. One pre-existing, CUDA-shared, dead-in-practice bug found and
worked around in the validation aid (PoolingOp CSE aliasing); not filed, not
a port defect, documented above for anyone else who writes a similar check.

## Validation 2026-08-11 (windows-gfx1151, AMD Radeon 8060S, gfx1151, RDNA3.5) -- PASS

Platform: AMD Radeon 8060S (gfx1151, RDNA3.5, integrated APU, wave32, warpSize=32), Windows 11 Enterprise 10.0.26100. HIP_VISIBLE_DEVICES=0. Compiler: TheRock all-clang ROCm 7.13.0a20260511 (pip-wheel, _rocm_sdk_core). Fork AMD-Ecosystem/marian-dev @ moat-port, head 1d0822bd (review-passed stage). This is additive evidence only -- windows gate already satisfied by windows-gfx1201 (completed at ba0ec806).

### Host setup

TheRock pip-wheel for gfx1151 (7.13.0a20260511) ships DLLs only -- no installed headers. Headers sourced from D:/Develop/TheRock/rocm-libraries/projects/ source tree. Stub cmake packages created in agent_space/marian-hip-cmake/ for HIP, hipblas, hipblaslt, hipsparse, hiprand, rocthrust, MIOpen, openblas; all throwaway (not committed to fork). Import .lib files generated from TheRock DLLs using llvm-readobj + MSVC lib.exe.

Sentencepiece submodule pin f006008f unfetchable; checked out master branch (1ca221c). Throwaway edits applied for Windows clang build (sentencepiece fPIC flag guard, constexpr kAnyType; faiss SSE header). All reverted before completion; fork is clean (git status --porcelain empty).

Additional linker fix: clang_rt.builtins-x86_64.lib added via CMAKE_EXE_LINKER_FLAGS (path: _rocm_sdk_core/lib/llvm/lib/clang/23/lib/windows/clang_rt.builtins-x86_64.lib) to supply __truncsfhf2/__extendhfsf2 fp16 soft-float builtins that HIP cmake strips by linking with -nostartfiles -nostdlib.

### Build

277/277 targets, EXIT:0. Multiple configure rounds needed to resolve header/library gaps in the TheRock pip-wheel-only setup (no installed headers). Build capped at -j6 to avoid APU thermal trips.

```
cmake -B agent_space/marian-build-gfx1151 -S projects/marian-dev/src \
  -DCMAKE_HIP_COMPILER=.../clang++ \
  -DCMAKE_PREFIX_PATH=agent_space/marian-hip-cmake \
  -DUSE_HIP=ON -DUSE_SENTENCEPIECE=ON -DUSE_FAISS=ON \
  -DUSE_FBGEMM=OFF -DUSE_TCMALLOC=OFF -DUSE_STATIC_LIBS=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_EXE_LINKER_FLAGS="<clang_rt.builtins-x86_64.lib>"
cmake --build agent_space/marian-build-gfx1151 -j6
```

### Runtime env (all tests)

```
SP=D:/Develop/TheRock/.venv/Lib/site-packages
ROCBLAS_USE_HIPBLASLT=0
ROCBLAS_TENSILE_LIBPATH=$SP/_rocm_sdk_libraries_gfx1151/bin/rocblas/library
HIPBLASLT_TENSILE_LIBPATH=$SP/_rocm_sdk_libraries_gfx1151/bin/hipblaslt/library
HIP_VISIBLE_DEVICES=0
# DLLs (24 total) copied next to each exe; must cd to exe dir before running
```

ROCBLAS_USE_HIPBLASLT=0 required: TheRock FP8 is_inf crash in libhipblaslt.dll (same as gfx1201/gfx1101).

### Unit test results

- run_binary_tests.exe: 9/9 PASS
- run_fastopt_tests.exe: 23/23 PASS
- run_utils_tests.exe: 8/8 PASS
- run_graph_tests.exe: 10/10 PASS -- GPU dispatch confirmed on gfx1151
- run_attention_tests.exe "*Attention (gpu)": 2/2 PASS
- run_transformer_tests.exe "*(gpu) fp32": 1/1 PASS; fp16 ABORTS "Broken type float16" (pre-existing Windows limitation: types.h _MSC_VER guard stubs float16 -- identical to gfx1201 and gfx1101)
- run_operator_tests.exe "Expression graph supports basic math operations (gpu)": 202/202 PASS (fp32); fp16 ABORTS pre-existing; "Compare aggregate operator": 1/1 PASS; CPU ops: 198/198 PASS
- run_rnn_tests.exe: ABORT "Broken type float16" (pre-existing Windows limitation, same as all other Windows arches)
- test_prod.exe: PASS (GPU GEMM correctness)

### Training smoke (GPU)

```
marian.exe --type transformer -t train.src train.tgt -m model.npz \
  --vocabs vocab.src.yml vocab.tgt.yml --dim-emb 64 --transformer-dim-ffn 128 \
  --transformer-heads 2 --enc-depth 2 --dec-depth 2 --after 100u --devices 0 \
  --shuffle-in-ram --tempdir tmp
```

Training completed successfully (100u, 1000-sentence corpus). RC:0. GPU used for all forward+backward passes.

### e2e decode (rocBLAS grouped-batched GEMM -- key gate)

gfx1101 (RDNA3 sibling) validation-failed here with hipErrorInvalidImage because gfx1101's rocBLAS Tensile library contained NO ELF .co kernel files for the FP32 batched (Alik_Bljk) layout, forcing a SPIR-V generic fallback that fails on Windows kpack_load_code_object.

gfx1151 Tensile library (150 files) includes 16 FP32 (SS) kernel files AND 20 ELF .co code objects covering Alik_Bljk_Cijk_Dijk (the exact layout hipblasGemmBatchedEx selects for marian's ProdBatched). Confirmed by listing gfx1151 library:
- TensileLibrary_Type_SS_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx1151.co (present)
- No SPIR-V fallback needed

marian-decoder.exe decoded successfully with:
- Small toy model (dim-emb=64, 20-word vocab, beam-size=6): RC:0
- Production-size model (dim-emb=512, 6-layer, 22K vocab, beam-size=6, mini-batch=32): RC:0, total time 0.44s

No hipErrorInvalidImage observed on gfx1151. The rocBLAS GB-GEMM path is fully operational on gfx1151.

### Verdict: completed -- windows-gfx1151

All GPU unit tests pass (fp32 path; fp16 abort is pre-existing Windows/types.h limitation, not a port bug). Training and e2e decode work correctly on real gfx1151 GPU. The rocBLAS grouped-batched GEMM blocker that failed gfx1101 does NOT affect gfx1151 -- the TheRock gfx1151 Tensile library ships proper ELF .co kernels for the SS (FP32) batched GEMM layout that marian's ProdBatched uses.

## 2026-08-12: upstream PR body replaced and marked ready for review

marian-nmt/marian-dev#1043 is OPEN and no longer a draft, at head `1d0822bd`
(unchanged; no code was pushed for this transition). Jeff Daily approved the
replacement body in session and ran both upstream writes himself -- the guard
shim refuses upstream `gh` writes from an agent, and the trusted publisher only
opens new PRs, so neither `gh pr edit` nor `gh pr ready` was agent-executed.

The draft comment of 2026-08-10 promised the three scope-outs before marking
ready; all three are in the branch and validated, so that promise is kept.

### Why the old body had to go

It still carried a `## Deliberate scope limits` section stating that sparse
matmul, collectives and cuDNN conv/pool were not ported -- the exact three
things the maintainer sent the port back for, all now implemented -- plus
"operator suite passes 284 of 287" and a question asking whether to guard the
csr-dot failures off. Marking ready with that text would have argued against
its own diff and re-opened the rejected scope.

### The body now on the PR

Source of truth is `agent_space/marian-dev-pr-body.md` (gitignored scratch;
the posted body is the durable copy). 12072 chars, verified byte-identical to
the file after posting. Ten sections: build integration, source changes, the
topk warp-synchronous tail, one section each for sparse / collectives /
conv-pooling, `## Three fixes the NVIDIA build needs too`, validation, known
limitations, build recipe.

Kept deliberately, per Jeff's ruling: the two shared upstream fixes stay
bundled in this PR AND keep their own section, because `USE_CUDNN=ON` neither
links nor produces correct shapes without them, so the requested MIOpen work
does not function otherwise. They are `options.cpp`'s missing
`Get<std::pair<int, int>>` instantiation and `PoolingOp` taking its input's
shape instead of the pooled one; the third item in that section is the sparse
zero-scratch guard, inert on cuSPARSE.

Corrections made while drafting, both of which were claims about upstream code:
- `fastopt.cpp` already had `As<std::pair<int, int>>` on master. Only
  `options.cpp`'s `Get<>` was missing, and it is the only one this branch adds.
  Verified with `git diff master..moat-port -- src/common/options.cpp
  src/common/fastopt.cpp`.
- `USE_CUDNN` defaults OFF and `USE_NCCL` defaults ON upstream
  (`CMakeLists.txt:33,38`); this branch changes neither default. The body says
  so rather than claiming both are on.

Two rounds of Jeff's wording edits: dropped a "neither of them the guess that
seemed likely at first" history sentence from the sparse section, and reworded
two spots that leaked internal gate vocabulary ("the CUDA no-regression check"
-> a statement about cuDNN 8 having removed the algorithm-selection API, and
the validation bullet -> "The CUDA build is unaffected"). LESSON: an
upstream-visible body must describe the change, not the investigation that
produced it, and must not name our own gates. `jargon.py` catches neither --
"no-regression" and narrative framing are both clean by that checker.

### Deferral rulings carried into the body (Jeff, 2026-08-12)

- `marian-topk-cub-segsort`: defer, ship as disclosed. Listed under known
  limitations as the unported CUB segmented-sort path, tests-only.
- `marian-gfx1101-batched-gemm-spirv`: defer, ship as disclosed. Listed as an
  external library gap with gfx1201/gfx1151 unaffected.
- `marian-poolingop-cse-alias`: now -- file upstream as its own issue, not
  mentioned in this PR. No existing marian-nmt/marian-dev issue covers it
  (searched pooling, layer_pooling, subexpression, avg_pooling; only unrelated
  #101 and #475). Draft at `agent_space/marian-dev-issue-pooling-cse.md`,
  awaiting a person to file it; still under discussion.

### State note

`pr_approval` stays empty and `review_pr` still points at CLOSED fork PR #1 at
`ba0ec806`. That is not repairable and does not need to be: the recorded
approval exists to let the unattended publisher prove what it is about to open,
and this PR was already open, so nothing reads it for this project.

## 2026-08-12: pooling/convolution CSE defect reported upstream as #1044

marian-nmt/marian-dev#1044, filed by Jeff Daily (agents cannot write upstream;
the guard shim refuses it and the trusted publisher only opens PRs). Body is
`agent_space/marian-dev-issue-pooling-cse.md`, verified byte-identical to the
issue after posting. Deliberately NOT mentioned in PR #1043: it is a
pre-existing upstream defect that affects CUDA identically and has nothing to
do with AMD support.

### What the report actually says, which is broader than what we first recorded

The framing that survived is not "pooling aliases avg to max" but "three node
classes are missing the `equal()`/`hash()` overrides the rest of the graph
consistently provides". `ExpressionGraph::add()` -> `findOrRemember` keys CSE on
`Node::hash()`/`Node::equal()`, which consider only name, `type()` string, value
type and child ids (`src/graph/node.h`), so a class carrying result-determining
state in a member must override both. Sixteen classes in
`node_operators_unary.h` alone do; `ConvolutionOp`, `PoolingOp` and
`PoolingWithMaskingOp` do not. Ranked by real exposure:

- `PoolingOp`: OBSERVED (gfx90a MIOpen check, 2026-08-11). `type()` is
  `"layer_pooling"` for every mode and the window/mode live in
  `PoolingWrapper`, so avg and max on one input in one graph alias. Got
  `5 4 8 7`, wanted `3 3 5.5 6.5`.
- `PoolingWithMaskingOp`: same missing overrides and it sits AFTER the
  `#endif`, so it is compiled in every build, cuDNN or not. Shares the
  `"layer_pooling"` string with `PoolingOp`, keeps `mask_` as a plain member
  rather than a child, and derives `shape_` from `width_` -- so a caller can
  get a node whose shape does not match its arguments. NOT run.
- `ConvolutionOp`: paddings/strides in `conv_`, no overrides, but kernel and
  bias ARE children, so only same-weights-different-padding would alias. Least
  exposed. NOT run.

Nothing in the tree reaches any of it: `avg_pooling` has no callers,
`max_pooling` has one (`examples/mnist/model_lenet.h`) on distinct inputs, and
`CharConvPooling` gives each convolution its own prefix and each pooling its own
convolution output. Reported as a trap for the next model or test, not as a live
wrong-output bug, and the report says which claims were run and which were read.

The report also offers why this group is the exception: two of the three cannot
be built, because `USE_CUDNN=ON` does not link without the
`Get<std::pair<int, int>>` instantiation that PR #1043 adds. No patch offer was
made (Jeff's call); the three candidate fix shapes are described and the choice
left to the maintainers, including that folding `mask_` into `equal()`/`hash()`
means `mask_->getId()` or promoting the mask to a real child, which reaches
`src/onnx/expression_graph_onnx_serialization.cpp`.

### Method note worth reusing

The first draft was scoped to the one observed symptom. Checking the claim
against upstream `master` before filing -- rather than trusting our own earlier
note -- is what turned up the other two classes and the "deviates from its own
convention" framing, which is the part that makes the report actionable. Verify
a bug report against the pristine upstream tree, not against the port branch or
against the note that recorded it.

### Search before filing

`gh search issues --repo marian-nmt/marian-dev` for pooling, layer_pooling,
subexpression, avg_pooling: no existing coverage (only unrelated #101, #475).
Note `gh search issues` takes `--state open|closed`, not `all`.
