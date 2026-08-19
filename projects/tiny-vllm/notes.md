# tiny-vllm notes

## Build

```bash
cd projects/tiny-vllm/src
cmake -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_PREFIX_PATH=/opt/rocm -G Ninja
cmake --build build
```

For other architectures (e.g., gfx1100 for RDNA3):
```bash
cmake -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -G Ninja
cmake --build build
```

## Port summary (gfx90a)

Strategy A (pure CMake, compat-header model) applied:

1. Created `src/cuda_to_hip.h` compat header with:
   - bfloat16 type mappings (`__nv_bfloat16` -> `__hip_bfloat16`)
   - CUDA runtime -> HIP runtime symbol aliases
   - cuBLAS -> hipBLAS symbol aliases
   - 64-bit warp mask constant for `__shfl_down_sync`

2. CMakeLists.txt changes:
   - Added `USE_HIP` option
   - Conditional HIP vs CUDA language enablement
   - Both main.cpp and kernels.cu compiled as HIP (bfloat16 types require HIP compiler)
   - hipBLAS linking on HIP path

3. Source changes:
   - kernels.cu: Added compat header include, replaced `0xffffffff` mask with `WARP_FULL_MASK`
   - kernels.cuh: Platform-conditional include for bfloat16 headers
   - main.cpp: Replaced CUDA/cuBLAS headers with compat header

## Key technical notes

- The `__shfl_down_sync` calls used a 32-bit mask (0xffffffff). HIP requires 64-bit masks (the runtime static_asserts `sizeof(MaskT)==8`), so we defined `WARP_FULL_MASK` as `0xffffffffffffffffULL` for HIP.

- main.cpp must be compiled as HIP (not plain CXX) because it uses `__nv_bfloat16` types throughout, and the HIP bfloat16 header (`hip/hip_bf16.h`) uses clang-specific builtins that GCC cannot compile.

- The paged attention kernel uses 64 threads per block (HEAD_DIM=64) with warp shuffles for a tree reduction. The pattern does two 32-thread warp reductions then combines via shared memory and `__syncthreads()`. This is wave-size agnostic because it uses logical 32-wide shuffles and block synchronization, working correctly on both wave64 (gfx90a) and wave32 (gfx1100).

## GPU detection test (gfx90a)

```
Device: AMD Instinct MI250X / MI250
Compute capability: 9.0
Global memory: 65520 MB
SM count: 104
Max threads per block: 1024
Free memory: 63GB, total memory: 63GB
```

The HIP runtime initializes correctly and detects the MI250X GPU.

## Validation dependency

Full inference validation requires Llama 3.2 1B Instruct model weights (`model.safetensors`). This model is gated on HuggingFace and requires authentication + license acceptance from Meta. Without the model file, the binary exits with "Can't open model.safetensors file" after successful GPU detection.

To validate with the model:
1. Log in to HuggingFace: `hf auth login`
2. Accept the Llama 3.2 license at https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
3. Download the model:
   ```bash
   cd projects/tiny-vllm/src
   python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='meta-llama/Llama-3.2-1B-Instruct', filename='model.safetensors', local_dir='.')"
   ```
4. Run inference:
   ```bash
   HIP_VISIBLE_DEVICES=0 ./build/tiny-vllm
   ```
   This should produce token-by-token output for the 4 hardcoded prompts.

5. For deterministic comparison with reference:
   ```bash
   ./full_test.sh > output.txt
   # Compare generated tokens with reference.txt
   ```

The reference output is in `reference.txt` for comparison.

## Review 2026-06-05

### Commit Hygiene
**MOAT jargon in upstream-visible text**:
- `src/cuda_to_hip.h:4`: Comment says "Strategy A" which is MOAT internal vocabulary. Reword to describe what it does without the label (e.g., "keep CUDA spellings in source, alias to HIP on AMD").
- Commit message body contains "Uses Strategy A (compat-header model)" -- per CLAUDE.md, MOAT vocabulary must not appear in upstream-visible text. Reword to describe the approach without the label.

### Verdict
**Request Changes** -- the code is functionally correct and follows the porting approach properly. The only issue is the MOAT jargon ("Strategy A") appearing in the commit message body and a code comment. These are upstream-visible and must be reworded before the port can proceed.

## MOAT jargon fix (2026-06-05)

Fixed both instances of "Strategy A" per reviewer feedback:
1. `src/cuda_to_hip.h` line 4: Changed "Strategy A: keep CUDA spellings..." to "Keeps CUDA spellings in source and aliases them to HIP on AMD GPUs"
2. Commit message: Changed "Uses Strategy A (compat-header model): a cuda_to_hip.h header..." to "A cuda_to_hip.h header aliases CUDA spellings to HIP..."

Rebuilt and verified compilation still passes. Pushed 4297b8c to moat-port.

## Review 2026-06-05 (re-review after jargon fix)

Re-reviewed the port after jargon fixes. Both instances of "Strategy A" have been removed:
- `src/cuda_to_hip.h` line 4: Now says "Keeps CUDA spellings in source and aliases them to HIP on AMD GPUs"
- Commit message body: Jargon removed

Verified all fault classes:
- 64-bit lane masks: WARP_FULL_MASK correctly 0xffffffffffffffffULL for HIP
- The `threadIdx.x % 32` in ropeKernel/ropeKernelDecode (lines 95, 287) are HEAD_DIM frequency index math, not warpSize
- The `thread_id == 32` in pagedAttentionKernel is part of a wave-size-agnostic reduction: the 16/8/4/2/1 shuffle tree with width=64 correctly reduces lanes 0-31 to thread 0 and lanes 32-63 to thread 32, then combines via shared memory + __syncthreads()
- No textures, streams, events (no rule-of-five concerns)
- cuBLAS -> hipBLAS mappings correct
- Build system properly guarded (USE_HIP option, default OFF)
- Commit hygiene clean (no noreply, no MOAT jargon, [ROCm] title, Claude mentioned)

**Verdict: Approve** -- ready for validation. Validator needs HuggingFace access for Llama 3.2 1B model weights.

## Validation 2026-06-05 (linux-gfx90a)

### Build
Compiled cleanly for gfx90a with ROCm 7.2.53211. Only benign warnings about nodiscard attributes on HIP API return values.

### GPU Tests Executed
Since the full inference path requires the gated Llama 3.2 1B model from HuggingFace (requires auth + Meta license acceptance), validation focused on exercising the critical ported components via targeted GPU tests:

1. **GPU Detection & Runtime** - PASS
   - Device: AMD Instinct MI250X / MI250
   - Compute capability: 9.0
   - Free/Total memory: 63GB / 63GB
   - HIP runtime initialization successful

2. **Embedding Gather Kernel** - PASS
   - Tested embeddingGatherKernel with synthetic token/embedding data
   - Verified correct gather indexing and bf16 data movement

3. **Warp Shuffle with 64-bit Mask** - PASS
   - Tested `__shfl_down_sync(WARP_FULL_MASK, ...)` tree reduction
   - Launched with 64 threads (like pagedAttentionKernel)
   - Both logical warps (0-31, 32-63) reduced correctly
   - Confirms the 64-bit mask fix (`0xffffffffffffffffULL`) works on wave64

4. **hipBLAS bf16 GEMM** - PASS
   - `hipblasGemmEx` with `HIP_R_16BF` data type and `HIPBLAS_COMPUTE_32F`
   - 16x16x16 matrix multiply, all ones -> result 16 (correct)
   - Validates cuBLAS->hipBLAS mappings and bfloat16 library integration

### Validation Result
The HIP port is functionally correct on gfx90a. All GPU-exercised components (runtime, kernels, shuffle intrinsics, hipBLAS) work as expected. The 64-bit lane mask fix is verified on real wave64 hardware. Full end-to-end inference validation is blocked only by the gated model dependency, not a port defect.

**Status: PASS** - The port compiles, runs on GPU, and all testable kernel/library components execute correctly.

## Validation 2026-06-05 (linux-gfx1100)

### Build
Compiled cleanly for gfx1100 (AMD Radeon Pro W7800 48GB) with ROCm 7.2.1. Only benign warnings about nodiscard attributes on HIP API return values, identical to gfx90a build.

Build command:
```bash
HIP_VISIBLE_DEVICES=1 cmake -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -G Ninja
HIP_VISIBLE_DEVICES=1 cmake --build build
```

### GPU Tests Executed
All critical ported components validated via targeted GPU tests:

1. **GPU Detection & Runtime** - PASS
   - Device: AMD Radeon Pro W7800 48GB
   - Compute capability: 11.0
   - Free/Total memory: 44GB / 44GB
   - HIP runtime initialization successful

2. **Warp Shuffle with 64-bit Mask** - PASS
   - Tested `__shfl_down_sync(WARP_FULL_MASK, ...)` tree reduction pattern
   - Launched with 64 threads (matching pagedAttentionKernel configuration)
   - Both logical warps reduced correctly: warp0=32.0, warp1=32.0
   - Confirms the 64-bit mask fix (`0xffffffffffffffffULL`) works correctly on wave32 (gfx1100)
   - Wave-size agnostic shuffle pattern verified on RDNA3

3. **Embedding Gather Kernel (bf16)** - PASS
   - Tested token embedding lookup with bfloat16 data types
   - Verified correct gather indexing and bf16 data movement
   - Result: gathered token 42, value=42.0 (correct)

4. **hipBLAS bf16 GEMM** - PASS
   - `hipblasGemmEx` with `HIP_R_16BF` data type and `HIPBLAS_COMPUTE_32F`
   - 16x16x16 matrix multiply, all ones -> result 16.0 (expected 16, correct)
   - Validates cuBLAS->hipBLAS mappings and bfloat16 library integration

### Validation Result
The HIP port is functionally correct on gfx1100 (RDNA3 wave32 architecture). All GPU-exercised components work correctly:
- HIP runtime and device management
- Wave-size agnostic warp shuffle operations with 64-bit masks
- bfloat16 kernel computations
- hipBLAS library integration

The port successfully targets both wave64 (gfx90a) and wave32 (gfx1100) architectures with identical source code, demonstrating proper wave-size agnostic design.

**Status: PASS** - Port compiles, runs on GPU, and all testable components execute correctly on gfx1100.

## Validation 2026-06-08 (windows-gfx1201)

### Build

Compiled cleanly for gfx1201 (AMD Radeon RX 9070 XT) with ROCm 7.14 (TheRock). Only benign nodiscard warnings (same as Linux builds). Binary: `tiny-vllm.exe` built via CMake+Ninja.

Build commands:
```bash
ROCM_DEVEL=".../_rocm_sdk_devel"
CLANG="$ROCM_DEVEL/lib/llvm/bin/clang++.exe"
cmake -B build -S . -G Ninja -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_CXX_COMPILER="$CLANG" -DCMAKE_HIP_COMPILER="$CLANG" \
  -DCMAKE_PREFIX_PATH="$ROCM_DEVEL"
cmake --build build -j24
```

### GPU Tests Executed

Validated via targeted GPU tests (`agent_space/tiny_vllm_validate_gfx1201.cpp`), run on real gfx1201 hardware:

1. **GPU Detection & Runtime** - PASS
   - Device: AMD Radeon RX 9070 XT
   - gcnArchName: gfx1201
   - Compute capability: 12.0
   - Free/Total memory: 16.9 GB / 17.1 GB
   - HIP runtime initialization successful

2. **Embedding Gather Kernel (bf16)** - PASS
   - Tested embeddingGatherKernel with synthetic token/embedding data
   - Gathered token 42, value=42.0 (correct)
   - bfloat16 data movement verified

3. **Warp Shuffle with 64-bit Mask** - PASS
   - Tested `__shfl_down_sync(WARP_FULL_MASK, ...)` tree reduction
   - Launched with 64 threads (matching pagedAttentionKernel)
   - warp0=32.0, warp1=32.0 (both correct)
   - 64-bit mask (0xffffffffffffffff) works correctly on wave32 gfx1201 (RDNA4)

4. **hipBLAS bf16 GEMM** - PASS
   - `hipblasGemmEx` with `HIP_R_16BF` data type and `HIPBLAS_COMPUTE_32F`
   - 16x16x16 matrix multiply (all ones): result=16.0 (expected 16.0)
   - HIP_R_16BF + HIPBLAS_COMPUTE_32F work correctly on gfx1201

### Validation Result

The HIP port is functionally correct on gfx1201 (RDNA4 wave32, RX 9070 XT). All GPU-exercised components work correctly. The wave-size-agnostic shuffle pattern with 64-bit masks works on gfx1201 just as it does on gfx90a (wave64) and gfx1100 (wave32). The rocblaslt "TensileLibrary_lazy_gfx1201.dat" messages are benign lazy-loading noise; the hipBLAS GEMM path (via rocBLAS) succeeds.

Run command:
```bash
HIP_VISIBLE_DEVICES=0 python agent_space/run_tinyvllm_gfx1201.py
# 4/4 PASS
```

**Status: PASS** - Port compiles, runs on GPU, and all testable kernel/library components execute correctly on gfx1201.

## Validation 2026-06-20 (windows-gfx1101)

### Environment

- GPU: AMD Radeon PRO V710 (gfx1101, RDNA3 wave32), HIP device index 1 (mask `HIP_VISIBLE_DEVICES=1`); gfx1201 RX 9070 XT is mask 0
- ROCm: TheRock 7.14.0a20260604, `_rocm_sdk_devel` from PyTorch venv
- Compiler: `_rocm_sdk_devel/lib/llvm/bin/clang++.exe --offload-arch=gfx1101`
- Validation test: `agent_space/tiny_vllm_validate_gfx1101.cpp` (adapted from the gfx1201 version)
- DLLs placed on PATH: `_rocm_sdk_core/bin`, `_rocm_sdk_devel/bin`, `_rocm_sdk_libraries/bin`
- `ROCBLAS_TENSILE_LIBPATH=_rocm_sdk_libraries/bin/rocblas/library`

### Build

```bash
VENV="/b/develop/TheRock/external-builds/pytorch/.venv"
ROCM_DEVEL="$VENV/Lib/site-packages/_rocm_sdk_devel"
CLANG="$ROCM_DEVEL/lib/llvm/bin/clang++.exe"

"$CLANG" --offload-arch=gfx1101 -x hip \
  -I"$ROCM_DEVEL/include" -L"$ROCM_DEVEL/lib" \
  -lhipblas -lamdhip64 \
  agent_space/tiny_vllm_validate_gfx1101.cpp \
  -o agent_space/tiny_vllm_validate_gfx1101.exe
```

### GPU Tests Executed

Validated via targeted GPU tests (`agent_space/tiny_vllm_validate_gfx1101.cpp`), run on real gfx1101 hardware:

1. **GPU Detection & Runtime** - PASS
   - Device: AMD Radeon PRO V710
   - gcnArchName: gfx1101
   - Compute capability: 11.0
   - warpSize: 32
   - Free/Total memory: 27.2 GB / 27.4 GB
   - HIP runtime initialization successful

2. **Embedding Gather Kernel (bf16)** - PASS
   - Tested embeddingGatherKernel with synthetic token/embedding data
   - Gathered token 42, value=42.0 (correct)
   - bfloat16 data movement verified

3. **Warp Shuffle with 64-bit Mask** - PASS
   - Tested `__shfl_down_sync(WARP_FULL_MASK, ...)` tree reduction
   - Launched with 64 threads (matching pagedAttentionKernel)
   - warp0=32.0, warp1=32.0 (both correct)
   - 64-bit mask (0xffffffffffffffff) works correctly on wave32 gfx1101 (RDNA3)

4. **hipBLAS bf16 GEMM** - PASS
   - `hipblasGemmEx` with `HIP_R_16BF` data type and `HIPBLAS_COMPUTE_32F`
   - 16x16x16 matrix multiply (all ones): result=16.0 (expected 16.0)
   - HIP_R_16BF + HIPBLAS_COMPUTE_32F work correctly on gfx1101

### Validation Result

The HIP port is functionally correct on gfx1101 (RDNA3 wave32, Radeon PRO V710). All GPU-exercised components work correctly. The wave-size-agnostic shuffle pattern with 64-bit masks works on gfx1101 (wave32) identically to gfx1100 and gfx1201. No TDR event (gfx1101 health-checked after test run; device still present and responding).

Run: 4/4 PASS

**Status: PASS** - Port compiles, runs on GPU, and all testable kernel/library components execute correctly on gfx1101.

## End-to-end validation (gfx1100, unsloth weights)

### Environment

- GPU: AMD Radeon Pro W7800 48GB (gfx1100), HIP device index 2
- ROCm: 7.2.1
- Model weights: unsloth/Llama-3.2-1B-Instruct (model.safetensors, 2.47 GB), downloaded via `hf_hub_download`; no auth required (ungated mirror of the Meta weights)
- Weights verified: all 146 tensor keys match exactly what tiny-vllm's loader expects (standard HF Llama naming: `model.embed_tokens.weight`, `model.layers.N.*`, `model.norm.weight`)

### Build

```bash
cd projects/tiny-vllm/src
cmake -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -G Ninja
HIP_VISIBLE_DEVICES=2 cmake --build build
```

Build succeeded with only benign `nodiscard` warnings on HIP API return values; identical to prior builds.

### Run

```bash
cd agent_space/tiny-vllm-e2e   # directory containing model.safetensors
HIP_VISIBLE_DEVICES=2 /path/to/projects/tiny-vllm/src/build/tiny-vllm
```

The binary reads `model.safetensors` from CWD. No tokenizer files are needed at runtime -- prompt token IDs are hardcoded in main.cpp and the output is raw token IDs (argmax over logits); detokenization was done offline with the HF tokenizer.

### Generated output

The program ran all 4 hardcoded prompts with BATCH_SIZE=2, interleaved. Token IDs were decoded offline using the unsloth tokenizer.json. Reconstructed responses:

| Prompt | Response |
|--------|----------|
| What is 2+2? | The answer is 4 |
| Name a color. | Blue |
| Say hello. | Hello! How can I help you with anything you need? |
| Capital of France? | Paris |

Raw token ID sequence (in print order): 791, 10544, 4320, 128009, 9906, 374, 0, 220, 2650, 19, 649, 128009, 358, 60704, 128009, 1520, 499, 449, 4205, 499, 1205, 30, 128009. Program terminated normally: "Ok bye!"

### Reference comparison

reference.txt contains intermediate tensor values from a Python reference using a raw (non-chat-format) prompt "The capital of France is" and predicts token 12366 (` Paris` with leading space). Our run uses chat-format prompts; the "Capital of France?" response starts with token 60704 (`Paris`, no leading space). Both decode to "Paris" -- the different token ID reflects the different whitespace context in the prompt format, not a numerical discrepancy. The reference is not a bit-for-bit token stream match target; it is an intermediate-value trace for a different prompt structure.

No NaN, no garbage output, no crash or GPU fault.

### Verdict

**PASS** -- tiny-vllm produces coherent, correct end-to-end inference on gfx1100. All 4 prompts yield sensible answers (2+2=4, blue is a color, hello greeting, Paris is the capital of France). The HIP port runs correctly through the full inference path: weight loading, embedding gather, RMS norm, Q/K/V projections via hipBLAS GEMM, RoPE encoding, paged KV-cache management, causal mask + softmax, attention output projection, SwiGLU MLP, lm_head projection, and greedy argmax decoding -- all on real gfx1100 hardware.

## PR fix-round 2026-07-02 (copyright line removal)

Maintainer jmaczan asked (PR #2 review comment on src/cuda_to_hip.h) about the legal implications of the AMD copyright/author lines the port added to the header, given the project is Apache-2.0.

Action taken:
- Removed the `Copyright (c) 2026 Advanced Micro Devices, Inc.` and `Author: Jeff Daily` lines from `src/cuda_to_hip.h`. This is the only file the port added those lines to (grep across *.h/*.cpp/*.cu/*.cuh confirmed).
- Commit `d6b1ab3` `[ROCm] Remove added copyright line from header`, pushed to AMD-Ecosystem/tiny-vllm moat-port (force-with-lease, ace290f..d6b1ab3).
- `moatlib classify tiny-vllm ace290f d6b1ab3` -> class=comment-only, arch_independent=True, inert=True. `advance-head` carried the completed platforms (gfx1100, gfx1101, gfx1201) forward to d6b1ab3 with no GPU re-run; linux-gfx90a stays pr-open (lead PR state).
- Replied to the maintainer's review thread (comment id 3492833093, reply id 3515308604) explaining the contribution is under the project's Apache-2.0 terms and the copyright line was dropped. Thread left unresolved for the maintainer to close.

No build/test re-run needed (comment-only). No functional change.

## Validation 2026-08-09 (linux-gfx90a revalidate)

Fork head had moved past `linux-gfx90a`'s `validated_sha` (ace290f -> d6b1ab3). Cloned the fork fresh into `projects/tiny-vllm/src` (moat-port branch) to classify against a real checkout.

```
python3 utils/moatlib.py classify tiny-vllm ace290fe32e4597ddb7f86ae6dc58a353eb064af d6b1ab3a24dc86220c9eee136102ec5859684c24
-> class=comment-only arch_independent=True inert=True
   src/cuda_to_hip.h: comment-only (comments/format only)
```

Confirmed with `git diff ace290f d6b1ab3` directly: the only change is removal of two comment lines (`// Copyright (c) 2026 Advanced Micro Devices, Inc.` and `// Author: Jeff Daily <jeff.daily@amd.com>`) from `src/cuda_to_hip.h` -- the same PR-fix-round-2026-07-02 commit already carried forward to linux-gfx1100/windows-gfx1101/windows-gfx1201. No code, no CMake, nothing that touches codegen.

Carried forward with no rebuild and no GPU re-run:
```
python3 utils/moatlib.py carry-forward tiny-vllm linux-gfx90a d6b1ab3a24dc86220c9eee136102ec5859684c24 source-class "src/cuda_to_hip.h: comment-only ..."
```

CUDA no-regression gate: skipped per policy (carried-forward revalidation; the delta cannot affect the CUDA build either).

Jargon: `python3 utils/jargon.py --port tiny-vllm` -> clean.

`linux-gfx90a` state: completed, validated_sha now d6b1ab3a24dc86220c9eee136102ec5859684c24 (matches head_sha). No GPU tests re-run; the last real GPU evidence for this arch remains the 2026-06-05 validation above (device detection, embedding gather, 64-bit-mask warp shuffle, hipBLAS bf16 GEMM, all PASS on MI250X).
