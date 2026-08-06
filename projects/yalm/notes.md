# yalm notes

## Fork / branch
- Fork: https://github.com/AMD-Ecosystem/yalm (Actions disabled)
- Port branch: `moat-port` (off upstream `main`); fork `main` stays a clean mirror.
- HEAD: 311e1c39ebf5cfad02ca13c70aaf7f9942f39101

## Build (gfx90a)
The project builds with a hand-written `src/Makefile` driving nvcc/hipcc directly; there is no CMake. The HIP path is a `USE_HIP=1` branch added to the Makefile.

```
cd projects/yalm/src
export HIP_VISIBLE_DEVICES=2          # this host: GCD0/1 busy, use GCD2 only
make clean
make test USE_HIP=1 HIPARCH=gfx90a    # -> build/test (GPU validation binary)
make      USE_HIP=1 HIPARCH=gfx90a    # -> build/main (full model)
```
Followers reuse the same source with `HIPARCH=gfx1100` / `HIPARCH=gfx1151`; the arch is a make variable, no source edit and no new fork commit.

## Validation (real gfx90a, MI250X)
```
make test USE_HIP=1 HIPARCH=gfx90a && ./build/test   # prints "All tests passed"
```
`test_cuda_kernels()` compares matmul / mha (attn_dot, attn_softmax, att_mix) / ffn against a CPU gold at epsilon 1e-4, launching kernels directly (no hipGraph). `test_attn()` is the CPU regression guard. `AMD_LOG_LEVEL=3 ./build/test` confirms dispatch on gfx90a with 64-wide blocks.

## Wave64 fixes (USE_HIP-guarded; CUDA path byte-identical)
1. matmul_wide / fused_matmul_add_residuals launched `<<<rows/32, warpSize*32>>>`. The literal `32` is warps-per-block, not warp width; on wave64 the block is 64*32 = 2048 > 1024 cap -> launch fails. Fixed with `MATMUL_WPB` (16 on HIP, 32 on CUDA), used consistently in grid divisor and block dim. WPB must divide the model dims (4096, 32000) -- 16 does.
2. att_mix `__shared__ float shared0/1[32]` indexed by threadIdx.x, launched with tpb.x = warpSize (64) -> OOB on wave64. Sized arrays to 64 (max wavefront width).
3. The three standalone test entry points (mha_cuda/matmul_cuda/ffn_cuda) hardcoded `warp_size = 32` / `max_threads_per_block = 1024` and never call set_cuda_device; now query device attributes lazily (`query_warp_size`/`query_max_threads_per_block`).

## Gotchas / traps for followers
- ROCm 7.2.x `__shfl_*_sync` static_assert that the mask is a 64-bit integer; `FULL_MASK 0xffffffff` (32-bit) fails to compile. Use `0xffffffffffffffffULL` under USE_HIP (correct for wave64 anyway). CUDA keeps the 32-bit mask.
- Host C++ TUs include model.h -> cuda_to_hip.h -> `<hip/hip_runtime_api.h>`. g++ needs `-D__HIP_PLATFORM_AMD__` (hipcc sets it automatically for the .cu); both `USE_HIP` and `__HIP_PLATFORM_AMD__` are added to host CFLAGS under USE_HIP.
- hipcc-compiled .cu host stub is not PIE; g++ host objects are. Link with `-no-pie` (added to LDFLAGS under USE_HIP) to avoid `R_X86_64_32 ... can not be used when making a PIE object`.
- `hipStreamLegacy` exists in ROCm 7.2.1 (`(hipStream_t)1`); no fallback needed. The test path uses it; the full-model path uses hipGraph stream capture (not exercised by the test gate).
- `kernel_bench("matmul-wide")` is misleadingly named: it calls `matmul_cuda` (the `matmul` kernel), NOT `matmul_wide`, and asserts nothing -- "All tests passed" there is the unconditional main() print. The real WPB launch path is only hit by the full-model `_forward_cuda`; it was validated separately with a standalone probe (agent_space, throwaway) at `<<<d/16, 64*16>>>` matching a CPU reference.

## Secondary / not the gate
Full-model `./build/main model.yalm -d cuda -m perplexity` needs a converted model (offline `convert.py`, torch) and exercises hipGraph capture/replay. Not run here (no model staged); not required for the gate. If hipGraph replay misbehaves on a follower, a non-graph dispatch fallback is low effort since every kernel is already directly launchable (the test path proves it).

## Validation 2026-06-02 (validator, linux-gfx90a)

Result: PASS -- linux-gfx90a completed, validated_sha=311e1c39ebf5cfad02ca13c70aaf7f9942f39101.

GPU: AMD Instinct MI250X (GFX Version gfx90a), GCD2 (HIP_VISIBLE_DEVICES=2). ROCm/hipcc 7.2.53211.

Commands run (from /var/lib/jenkins/moat/projects/yalm/src):

```
export HIP_VISIBLE_DEVICES=2
make clean && make test USE_HIP=1 HIPARCH=gfx90a   # clean build; exit 0
./build/test                                         # run 1 -> "All tests passed"
./build/test                                         # run 2 -> "All tests passed" (deterministic)
```

Wrapped with `utils/timeit.sh yalm compile` and `utils/timeit.sh yalm test`.

gfx90a device dispatch confirmed via AMD_LOG_LEVEL=3:
```
hip_fatbin.cpp: Using native code object for device: amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack- co: amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-
```

Pass/fail counts: matmul/mha/ffn CPU-vs-GPU tests PASS (eps 1e-4); test_attn() CPU regression guard PASS. Total: 1 binary, all subtests pass, both runs identical.

Caveat carried from review: the WPB launch path (matmul_wide/fused_matmul_add_residuals) is not exercised by ./build/test -- it runs only in the full-model _forward_cuda, which requires a staged model (none downloaded). The porter/reviewer verified it via an agent_space probe at <<<d/16, 64*16>>> vs CPU reference. The formal gate is ./build/test + this documented caveat.

## Review 2026-06-02 (reviewer, linux-gfx90a)
Verdict: review-passed. Reviewed `git diff 6cd1ef6...311e1c3` on moat-port. Built clean from scratch (`make clean && make test USE_HIP=1 HIPARCH=gfx90a`, hipcc 7.2.53211, gfx90a) and ran `./build/test` on GCD2 -> "All tests passed". Confirmed gfx90a device code embedded (`roc-obj-ls`: hipv4-amdgcn-amd-amdhsa--gfx90a, 86888 bytes). Wave64 fixes verified as correct and the CUDA path preserved.

Verified correct:
- WPB self-consistency (infer.cu:29-31, 297-309, 314-329, 963/1003/1236/1242): `<<<rows/MATMUL_WPB, warp_size*MATMUL_WPB>>>` with WPB=16 on HIP gives 64*16=1024 <= cap; kernel-internal `blockDim.x/warpSize` == MATMUL_WPB == grid divisor, so blocktranspose store `block_start_i = blockIdx.x*blockDim.x/warpSize` tiles every row; 16 divides 4096 and 32000. CUDA WPB=32 keeps the original /32,*32 exactly.
- att_mix shared sizing (infer.cu:468-470, 557-565): shared0/1 indexed by threadIdx.x in [0,warpSize), sized to WARP_SIZE_MAX=64; head_dim=128 covered by `i=2*threadIdx.x; i<head_dim; i+=2*warpSize` (one pass at warpSize=64). blocktranspose sm[32] and block_all_reduce shared[32] hold <=16 wavefronts safely.
- Test entry points (infer.cu:1017/1061/1087) now use lazy query_warp_size()/query_max_threads_per_block() instead of hardcoded 32/1024; matches device-side warpSize=64.
- FULL_MASK widened to 0xffffffffffffffffULL only under USE_HIP (infer.cu:15-19); CUDA keeps 0xffffffff.
- Compat header (cuda_to_hip.h): libc-before-hip, host-safe (__HIPCC__ gates hip_runtime.h vs hip_runtime_api.h), aliases only used symbols; NVIDIA branch unchanged. model.h reroute correct.
- Makefile USE_HIP branch: hipcc -x hip --offload-arch, -lamdhip64 -no-pie, host CFLAGS get -DUSE_HIP -D__HIP_PLATFORM_AMD__; CUDA path untouched.
- Commit hygiene: title 60 chars `[ROCm] ...`, no noreply/Co-Authored/ghstack, mentions Claude, no em-dash. Fork main == upstream 6cd1ef6 (clean mirror); fork/moat-port == HEAD. Actions disabled (api enabled=false).
- No texture/surface, no library swaps, no per-arch hack in shared code (changes are wave-agnostic rewrites or USE_HIP-guarded).

Minor (non-blocking; porter may address opportunistically, not required to re-validate):
- att_mix shared array size changed 32 -> 64 unconditionally (infer.cu:468), so it is NOT byte-identical on the NVIDIA path (extra 256B shared, indices 32..63 unused, behavior unchanged). The commit message says the wave64 fixes are "guarded by USE_HIP only where the value genuinely differs" -- here the value differs but is unguarded. Harmless on CUDA; the claim is slightly imprecise.
- hipcc command places `-x hip` after the input file (Makefile CUFLAGS), so clang emits `warning: '-x hip' after last input file has no effect`. The .cu still compiles as HIP (verified gfx90a code object present) because clang treats .cu as HIP by default, so this is cosmetic; could move `-x hip` ahead of `$<` or drop it.
- hipGraph capture/replay path (_forward_cuda, add_or_update_kernel_node) is unverified by the gate (full-model only, no model staged). Already documented as secondary; flagging that gfx90a graph parity remains unproven for the eventual end-to-end check.

## Validation 2026-06-02 (gfx1100)

Result: PASS -- linux-gfx1100 completed, validated_sha=311e1c39ebf5cfad02ca13c70aaf7f9942f39101.

GPU: AMD Radeon Pro W7800 (gfx1100, RDNA3, wave32), HIP_VISIBLE_DEVICES=0. ROCm/hipcc 7.2.53211. Wavefront size: 32 (confirmed via `rocminfo`: "Wavefront Size: 32(0x20)"). Workgroup max: 1024.

Commands run (from /var/lib/jenkins/moat/projects/yalm/src):

```
make clean && make test USE_HIP=1 HIPARCH=gfx1100   # clean build; exit 0, ~4.3s
HIP_VISIBLE_DEVICES=0 ./build/test                   # run 1 -> "All tests passed"
HIP_VISIBLE_DEVICES=0 ./build/test                   # run 2 -> "All tests passed" (deterministic)
```

gfx1100 dispatch confirmed via AMD_LOG_LEVEL=3:
```
hip_fatbin.cpp: Using native code object for device: amdgcn-amd-amdhsa--gfx1100 co: amdgcn-amd-amdhsa--gfx1100
Gfx Major/Minor/Stepping: 11/0/0
```
roc-obj-ls: hipv4-amdgcn-amd-amdhsa--gfx1100, 85144 bytes.

Kernels dispatched (ShaderName log): matmul<float>, attn_dot, attn_softmax, att_mix, fused_ffn_w1_w3_glu_act<float, GELU>, matmul<float> (ffn w2). All five test-path kernels exercised.

Pass/fail counts: test_attn() CPU regression guard PASS; test_cuda_kernels() matmul/mha (attn_dot+attn_softmax+att_mix)/ffn CPU-vs-GPU epsilon 1e-4 comparisons all PASS; both runs identical. No HSA faults, no error output.

Wave32 verdict (warpSize=32 on gfx1100):

- MATMUL_WPB geometry: the test gate uses the simple `matmul` kernel (not matmul_wide), launched `<<<d, warpSize>>>` = `<<<16, 32>>>`. The MATMUL_WPB=16 path (matmul_wide/fused_matmul_add_residuals at `<<<rows/16, 32*16=512>>>`) is in the full-model _forward_cuda only and is not exercised by `./build/test` -- same documented caveat as gfx90a; the porter validated it via an agent_space probe at gfx90a geometry; at wave32 the block would be 32*16=512 <= 1024 cap, so no launch-geometry failure.

- att_mix shared arrays: `shared0/1[WARP_SIZE_MAX=64]` indexed by `threadIdx.x`. At wave32, `tpb.x=32`, so threadIdx.x in [0,31]; only slots 0..31 are accessed, slots 32..63 unused. No OOB, no stale read. Confirmed by the mha xout comparison passing at epsilon 1e-4.

- Warp reductions at width 32: `warp_reduce_sum` / `warp_all_reduce_max` / `warp_all_reduce_sum` loop `for offset = warpSize/2; offset > 0; offset /= 2` (device-side `warpSize`=32 at runtime), reducing 16->8->4->2->1 -- correct 5-step tree for 32-wide warps, no 64-lane assumption. `FULL_MASK=0xffffffffffffffffULL` with only 32 active lanes: on gfx1100 the low 32 bits are the active mask and the upper 32 are ignored by hardware, so the full-mask shuffle is valid. All reductions produce correct values (matmul/mha/ffn pass CPU gold at 1e-4).

- `block_all_reduce_max/sum` shared[32]: at most 32 warps per block (1024/32=32); shared capacity exactly matches. At block=32 threads (1 warp), path `if (blockDim.x < warpSize) return val` early-exits before the shared write, so no hazard.

- No 0x1016 (signal 11 / page fault), no wrong output, no NaN, no launch failure.

No source or fork changes required; the commit at 311e1c39ebf5 validates as-is on wave32. The `-x hip` cosmetic warning persists (noted in gfx90a review; non-blocking).

## Validation 2026-06-07 (linux-gfx90a, revalidate carry-forward)

Result: CARRY-FORWARD -- linux-gfx90a validated_sha advanced to 006a0fdfa796d1a4ea4625e9fbbc4b8ed25e739c via binary equivalence; no GPU re-run needed.

Delta: 006a0fdfa7 adds `host_ptr_to_device()` in `src/infer.cu` guarded by `#if defined(USE_HIP) && defined(_WIN32)`. On Linux `_WIN32` is not defined; the helper compiles to `return host;` and is inlined away. No device code change.

Verification: `python3 utils/codeobj_diff.py build_old/src/infer.cu.o build_new/src/infer.cu.o` -> `verdict=identical` (device ISA + exported symbols byte-identical). Both builds used `hipcc --offload-arch=gfx90a` (ROCm 7.2.53211).

arch: gfx90a (AMD Instinct MI250X), HIP_VISIBLE_DEVICES=3. No GPU execution performed.

## Validation 2026-06-07 (linux-gfx1100, revalidate carry-forward)

Result: CARRY-FORWARD -- linux-gfx1100 validated_sha advanced to 006a0fdfa796d1a4ea4625e9fbbc4b8ed25e739c via binary equivalence; no GPU re-run needed.

Delta: 006a0fdfa7 adds `host_ptr_to_device()` in `src/infer.cu` guarded by `#if defined(USE_HIP) && defined(_WIN32)`. On Linux `_WIN32` is not defined; the helper compiles to `return host;` and is inlined away. No device code change.

Verification: built at 311e1c39 and 006a0fd for gfx1100 (`make test USE_HIP=1 HIPARCH=gfx1100`), then `python3 utils/codeobj_diff.py lfs-old-gfx1100/infer.cu.o lfs-new-gfx1100/infer.cu.o` -> `verdict=identical` (device ISA + exported symbols byte-identical). Both builds used `hipcc --offload-arch=gfx1100` (ROCm 7.2.53211).

arch: gfx1100 (AMD Radeon Pro W7800, RDNA3, wave32). No GPU execution performed.

## Validation 2026-06-07 (windows-gfx1201)

Result: PASS -- windows-gfx1201 completed, validated_sha=006a0fdfa796d1a4ea4625e9fbbc4b8ed25e739c.

GPU: AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32). HIP_VISIBLE_DEVICES=0 (gfx1101 offline this session; gfx1201 shifted to device 0). ROCm 7.14.0a20260604 (TheRock, _rocm_sdk_devel, hipcc AMD clang 23.0, MSVC target).

### Windows delta commit (required, new on moat-port)

Added `host_ptr_to_device()` helper in `src/infer.cu` (guarded by `defined(USE_HIP) && defined(_WIN32)`) and applied it in the three test entry points (`mha_cuda`, `matmul_cuda`, `ffn_cuda`):

On Linux, `hipHostRegister` creates a UVA alias so the host pointer is directly GPU-addressable. On Windows (WDDM), it pins the allocation but does NOT alias it into GPU VA space; the kernel must use the device-side address returned by `hipHostGetDevicePointer`. Without this fix, all kernel outputs are zero (writes silently dropped). Linux and CUDA paths are unchanged (the `#else` branch returns host unchanged, compiling to identical code).

Fork commit: `006a0fdfa796d1a4ea4625e9fbbc4b8ed25e739c`

This commit requires linux-gfx90a and linux-gfx1100 to revalidate (the classifier sees `mixed` since it cannot distinguish the `_WIN32` guard from a behavior-affecting change). On Linux the compiled code is identical to 311e1c39 -- the `_WIN32` guard means the new function compiles to `return host;` and is inlined away -- so the Linux validators may use `codeobj_diff.py` to verify binary equivalence and carry forward without a GPU re-run.

### Build

No CMake; hand-built with hipcc (AMD clang 23.0, MSVC target, `--offload-arch=gfx1201`). POSIX compat shims (agent_space/yalm_win_compat: `sys/mman.h`, `unistd.h`) allow codec.cpp and test.cpp to compile; the POSIX paths (from_file, mem_bench) are dead code during the test run. fmt 11.0.2 `FMT_STRING` consteval is broken on AMD clang MSVC target with c++20 (pointer arithmetic in consteval context); compiled with `-std=c++17` instead (infer.cu's HIP kernels use the .cu path and are unaffected). TheRock DLLs copied to exe directory (amdhip64_7.dll, amd_comgr.dll, hiprtc0714.dll, hiprtc-builtins0714.dll, rocm_kpack.dll) so the exe-directory-search (#1) beats System32 (#2).

Build script: `agent_space/build_yalm_gfx1201.sh`

```
bash agent_space/build_yalm_gfx1201.sh
# -> projects/yalm/src/build/win/test.exe
```

### Test run

```
HIP_VISIBLE_DEVICES=0 projects/yalm/src/build/win/test.exe   # run 1 -> "All tests passed"
HIP_VISIBLE_DEVICES=0 projects/yalm/src/build/win/test.exe   # run 2 -> "All tests passed" (deterministic)
```

Kernels dispatched (palvirtual log): `_Z6matmulIfEvPKT_PKfiiPf`, `_Z8attn_dotPK6__halfPKfiiiiiPf`, `_Z12attn_softmaxPKfiiiPf`, `_Z7att_mixPK6__halfPKfiiiiiPf`, `_Z23fused_ffn_w1_w3_glu_actIfL14ActivationType0EEvPKT_S3_PKfiiPf`, `_Z6matmulIfEvPKT_PKfiiPf`. All 6 test-path kernels exercised. Code Object V5 (AOT, native gfx1201). `hipHostGetDevicePointer: Returned hipSuccess` for all output registrations.

Pass/fail counts: test_attn() CPU regression guard PASS; test_cuda_kernels() matmul/mha (attn_dot+attn_softmax+att_mix)/ffn CPU-vs-GPU epsilon 1e-4 comparisons all PASS; both runs identical.

## Validation 2026-06-16 (windows-gfx1101)

Result: PASS -- windows-gfx1101 completed, validated_sha=006a0fdfa796d1a4ea4625e9fbbc4b8ed25e739c.

GPU: AMD Radeon PRO V710 (gfx1101, RDNA3, wave32). HIP_VISIBLE_DEVICES=0. ROCm 7.14.0a20260604 (TheRock, _rocm_sdk_devel, hipcc AMD clang 23.0, MSVC target).

### Build

Identical to the gfx1201 build (agent_space/build_yalm_gfx1101.sh) except `--offload-arch=gfx1101` and output goes to `build/win_gfx1101/`. No source changes required; the same Windows delta commit (006a0fdfa7, host_ptr_to_device for WDDM) applies identically to gfx1101. TheRock DLLs copied to exe directory (amdhip64_7.dll, amd_comgr.dll, hiprtc0714.dll, hiprtc-builtins0714.dll, rocm_kpack.dll). gfx1101 AOT code object confirmed in binary (`amdgcn-amd-amdhsa--gfx1101` at offset 54538 in infer.cu.o).

Build script: `agent_space/build_yalm_gfx1101.sh`

```
bash agent_space/build_yalm_gfx1101.sh
# -> projects/yalm/src/build/win_gfx1101/test.exe
```

### Test run

```
HIP_VISIBLE_DEVICES=0 projects/yalm/src/build/win_gfx1101/test.exe   # run 1 -> "All tests passed"
HIP_VISIBLE_DEVICES=0 projects/yalm/src/build/win_gfx1101/test.exe   # run 2 -> "All tests passed" (deterministic)
```

`hipHostGetDevicePointer: Returned hipSuccess` for all output registrations (AMD_LOG_LEVEL=3 confirms TheRock amdhip64_7.dll loaded from exe dir, not System32).

Pass/fail counts: test_attn() CPU regression guard PASS; test_cuda_kernels() matmul/mha (attn_dot+attn_softmax+att_mix)/ffn CPU-vs-GPU epsilon 1e-4 comparisons all PASS; both runs identical.

Wave32 on gfx1101 (same RDNA3 family as gfx1100 Linux): warpSize=32, MATMUL_WPB=16 (block=512<=1024), att_mix shared[64] with threadIdx.x in [0,31] (slots 32..63 unused, no OOB), FULL_MASK upper 32 bits ignored by hardware. Matches the gfx1100 wave32 analysis in the Linux validation record.

## PR prep 2026-06-17

head_sha reconciled 006a0fd (stale) -> f10d84c. Real fork tip was 01f3291 (byte-scrub); reset to it before prep so the scrub was not lost.

Prep commit f10d84c (comment/doc-only, on top of 01f3291):
- README.md: added "Building for AMD GPUs (ROCm/HIP)" section (USE_HIP=1, HIPARCH select, ROCM_PATH override) next to the CUDA build block; parallel HIP note in the test section; widened the "NVIDIA GPU required" limitation to "NVIDIA (CUDA) or AMD (ROCm/HIP)".
- src/cuda_to_hip.h: added AMD copyright + author header (new file, we authored it); removed an internal cross-reference from a comment.
- Attribution decision: header added ONLY to the new shim. Makefile/infer.cu/model.h have no upstream copyright header, so a lone AMD line would falsely imply sole authorship -- not added.

advance-head classified the combined 01f3291 -> f10d84c delta comment-only; all completed platforms (linux-gfx90a, linux-gfx1100, windows-gfx1201, windows-gfx1101) carried forward to validated_sha=f10d84c with no GPU re-run. pr-ready=True (windows-gfx1151 non-viable/retired, scoped out of the PR claim). Compile-checked the HIP test build on gfx90a (ROCm 7.2.53211, exit 0, build/test produced) -- compile check, not a GPU re-validation.

## End-to-end validation (gfx1100, hipGraph path, Mistral-7B) 2026-06-17

**Verdict: PASS** -- full-model hipGraph capture/replay runs correctly end-to-end on gfx1100.

**GPU:** AMD Radeon Pro W7800 48GB (gfx1100, RDNA3, wave32). HIP_VISIBLE_DEVICES=0. ROCm/hipcc 7.2.53211 (AMD clang 22.0.0git, roc-7.2.1).

**Supplementary evidence for upstream PR** -- the kernel-test gate (./build/test, `All tests passed`) already closed the formal validation. This run confirms the hipGraph path (_forward_cuda, add_or_update_kernel_node) that the test binary does not exercise.

### Model fetch

```
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='mistralai/Mistral-7B-Instruct-v0.2',
    local_dir='agent_space/yalm-e2e/Mistral-7B-Instruct-v0.2',
    ignore_patterns=['*.bin', 'original/*'],
)
"
# -> 14GB (3 safetensors shards + config/tokenizer)
```

### Convert

```
cd projects/yalm/src
python3 convert.py --dtype fp16 \
    /var/lib/jenkins/moat/agent_space/yalm-e2e/mistral-7b-instruct-fp16.yalm \
    /var/lib/jenkins/moat/agent_space/yalm-e2e/Mistral-7B-Instruct-v0.2/
# -> 14GB .yalm file (226 tensors converted fp16, 32-layer Mistral config)
```

### Build

```
make -C projects/yalm/src USE_HIP=1 HIPARCH=gfx1100
# -> projects/yalm/src/build/main (6.3MB ELF); hipcc --offload-arch=gfx1100, -lamdhip64 -no-pie
```

### Run

Run 1 -- prompt "What is a large language model?", 100 steps:

```
HIP_VISIBLE_DEVICES=0 projects/yalm/src/build/main \
    agent_space/yalm-e2e/mistral-7b-instruct-fp16.yalm \
    -d cuda -m c -i "What is a large language model?" -n 100 -t 0.7
```

Generated text:

```
A large language model is a type of artificial intelligence (AI) model that is designed to
understand and generate human-like text. It is "large" because it has been trained on a vast
amount of data, enabling it to learn complex patterns and relationships in language. Large
language models can be used for a variety of natural language processing tasks, such as text
generation, translation, summarization, and question answering. Some popular large language
models include BERT, GPT-3, and
```

Stats: 108 tokens, throughput 26.667 tok/s, latency 0.0375 s/tok, hydrate 0.289 s, bandwidth 386.43 GB/s, total 4.05 s.

Run 2 -- prompt "Explain what ROCm is for GPU computing.", 60 steps:

```
HIP_VISIBLE_DEVICES=0 projects/yalm/src/build/main \
    agent_space/yalm-e2e/mistral-7b-instruct-fp16.yalm \
    -d cuda -m c -i "Explain what ROCm is for GPU computing." -n 60 -t 0.7
```

Generated text:

```
ROCm (Radeon Open Compute Platform) is an open-source software platform and programming model
developed by AMD for GPU computing, machine learning, and data center applications. ROCm
provides a complete software stack for developing, optimizing, and deploying applications on
AMD GP
```

Stats: 73 tokens, throughput 26.681 tok/s, latency 0.037479 s/tok, hydrate 0.473 s, bandwidth 386.58 GB/s, total 2.736 s.

### hipGraph path confirmed (AMD_LOG_LEVEL=3)

```
hip_fatbin.cpp: Using native code object for device: amdgcn-amd-amdhsa--gfx1100 co: amdgcn-amd-amdhsa--gfx1100
hip_graph.cpp: hipGraphInstantiate ( ... ) -> hipSuccess    # capture phase (first call per topology)
hip_graph.cpp: hipGraphLaunch ( ... )      -> hipSuccess    # replay every subsequent token step
hip_graph.cpp: hipGraphLaunch ( ... )      -> hipSuccess    # ... (one per generated token)
```

Sequence is: `hipStreamBeginCapture` -> kernel nodes added via `hipStreamGetCaptureInfo_v2` / `hipGraphAddKernelNode` -> `hipStreamEndCapture` -> `hipGraphInstantiate` -> `hipGraphLaunch` (repeated per step). The graph is instantiated twice (once for the prefill/FULL mode, once for the decode/GENERATE mode); replay holds for all subsequent steps within each mode. No HSA faults, no NaN, no wrong output. Both runs produce semantically correct, fluent text.

## Re-validation after WPB/host-ptr/WARP_SIZE_MAX fixes (gfx1100) 2026-06-17

Result: PASS -- linux-gfx1100 completed, validated_sha=3219ab7de814b25eb885ffe32f80b873b137a36c.

GPU: AMD Radeon Pro W7800 48GB (gfx1100, RDNA3, wave32). HIP_VISIBLE_DEVICES=0 (all 4 GPUs idle; index 0 chosen). ROCm/hipcc 7.2.53211.

Three functional fixes validated:
1. matmul_wpb now runtime-derived (`max_threads_per_block / warp_size`): yields 32 on wave32 (gfx1100), so the matmul launch geometry is `<<<rows/32, 32*32=1024>>>` -- back to the 1024-thread block the W7800 prefers. Both 16 (wave64) and 32 (wave32) divide 4096 and 32000.
2. host_ptr_to_device now always uses `hipHostGetDevicePointer` on HIP (not `_WIN32`-gated); `hipHostRegister` passes `hipHostRegisterMapped`. Confirmed by AMD_LOG_LEVEL=3: `hipHostRegister(..., 2)` -> `hipHostGetDevicePointer` -> `hipSuccess` for every test output buffer.
3. WARP_SIZE_MAX conditionalized: `#if USE_HIP 64 #else 32`. On gfx1100 (wave32) the shared arrays remain 64-wide (no regression; threadIdx.x in [0,31] so slots 32..63 unused).

### Part 1 -- Kernel-test gate

```
utils/timeit.sh yalm compile -- bash -c "make -C projects/yalm/src clean && make -C projects/yalm/src test USE_HIP=1 HIPARCH=gfx1100"
utils/timeit.sh yalm test -- bash -c "HIP_VISIBLE_DEVICES=0 projects/yalm/src/build/test && HIP_VISIBLE_DEVICES=0 projects/yalm/src/build/test"
```

Build: clean, exit 0. Only the cosmetic `'-x hip' after last input file has no effect` warning (pre-existing).
Run 1: "All tests passed"
Run 2: "All tests passed" (deterministic)

AMD_LOG_LEVEL=3 confirms:
- `hipHostRegister(..., 2)` -> `hipHostGetDevicePointer: Returned hipSuccess` (fix 2 active on Linux)
- `Using native code object for device: amdgcn-amd-amdhsa--gfx1100 co: amdgcn-amd-amdhsa--gfx1100`

### Part 2 -- End-to-end (hipGraph path, Mistral-7B)

Model: `agent_space/yalm-e2e/mistral-7b-instruct-fp16.yalm` (14GB, from prior e2e run -- reused).

```
make -C projects/yalm/src USE_HIP=1 HIPARCH=gfx1100    # -> build/main
utils/timeit.sh yalm e2e -- bash -c "HIP_VISIBLE_DEVICES=0 projects/yalm/src/build/main \
    agent_space/yalm-e2e/mistral-7b-instruct-fp16.yalm \
    -d cuda -m c -i 'What is a large language model?' -n 100 -t 0.7"
```

Generated text:
```
A large language model is a type of artificial intelligence (AI) model that can process and generate
human-like text based on the input it receives. These models are trained on vast amounts of text data,
allowing them to understand and generate coherent and contextually relevant responses to prompts. They
can be used for a variety of applications, such as text generation, translation, summarization, and
question answering. Some popular large language models include BERT, RoBERTa, GPT-
```

Stats: 108 tokens, throughput 26.654 tok/s, latency 0.037519 s/tok, hydrate 0.29 s, bandwidth 386.24 GB/s, total 4.052 s.

hipGraph confirmed (AMD_LOG_LEVEL=3):
- `hipGraphAddKernelNode: Returned hipSuccess` (multiple -- graph capture phase)
- `hipGraphInstantiate: Returned hipSuccess` (twice: prefill + decode graphs)
- `hipGraphLaunch: Returned hipSuccess` (repeated -- one per decode step)
- `Using native code object for device: amdgcn-amd-amdhsa--gfx1100`

No HSA faults, no NaN, no wrong output. Coherent, semantically correct text.

Note: linux-gfx90a and windows-gfx1201 remain in `revalidate` at validated_sha=f10d84c; they must re-validate 3219ab7 on their respective hosts. This run covers gfx1100 only.

## Correctness fixes at 3219ab7 (post-validation, functional -> revalidate)

Three functional fixes in src/infer.cu on top of f10d84c. These flip every previously-completed AMD platform to revalidate (correct -- they are not behavior-preserving).

- MATMUL_WPB was a compile-time constant (16 on HIP, 32 on CUDA). Replaced with a runtime global `matmul_wpb = max_threads_per_block / warp_size`, computed once in set_cuda_device. Yields 32 on wave32 (RDNA/CUDA) and 16 on wave64 (gfx90a). MUST be host-side: the launch grid `<<<rows/wpb, warp_size*wpb>>>` is configured on the host, and `__GFX9__` is device-only -- a host read of an arch macro is always wrong, and the old wave32->16 needlessly halved occupancy. Both 16 and 32 divide 4096 and 32000.
- host_ptr_to_device: was gated `USE_HIP && _WIN32`. The real cause is host-registered memory being device-addressable only through the MAPPED-flag mapping in the BAR aperture; a non-large-BAR Linux host faults. Now `hipHostGetDevicePointer` is used unconditionally on HIP, and register_cuda_host passes `hipHostRegisterMapped` on HIP (not in the shim, used as a raw hip symbol like hipHostGetDevicePointer). CUDA keeps cudaHostRegisterDefault + UVA no-op.
- WARP_SIZE_MAX (sizes att_mix __shared__ arrays): was unconditional 64, over-allocating CUDA's 32-wide shared arrays. Now `#if USE_HIP 64 #else 32`.

Compile-verified on this gfx1100 (wave32) host, ROCm 7.2.1: `make -C src clean && make -C src USE_HIP=1 HIPARCH=gfx1100` and the `test` target both build clean. NOT GPU-validated here; validator re-runs on real GPU per platform.

## Re-validation after WPB/host-ptr/WARP_SIZE_MAX fixes (gfx90a) 2026-06-17

Result: PASS -- linux-gfx90a completed, validated_sha=3219ab7de814b25eb885ffe32f80b873b137a36c.

GPU: AMD Instinct MI250X (GFX Version gfx90a), GCD2 (HIP_VISIBLE_DEVICES=2). ROCm/hipcc 7.2.53211.

Three functional fixes validated:
1. matmul_wpb now runtime-derived (`max_threads_per_block / warp_size`): yields 16 on wave64 (gfx90a), so the matmul launch geometry is `<<<rows/16, 64*16=1024>>>` -- back to the correct 1024-thread cap on MI250X.
2. host_ptr_to_device now always uses `hipHostGetDevicePointer` on HIP (not `_WIN32`-gated); `hipHostRegister` passes `hipHostRegisterMapped` flag (flag=2). Confirmed by AMD_LOG_LEVEL=3: `hipHostRegister(..., 2)` -> `hipHostGetDevicePointer: Returned hipSuccess` for every test output buffer.
3. WARP_SIZE_MAX conditionalized: `#if USE_HIP 64 #else 32`. On gfx90a (wave64) shared arrays remain 64-wide (correct).

Commands run (from /var/lib/jenkins/moat/projects/yalm/src):

```
export HIP_VISIBLE_DEVICES=2
utils/timeit.sh yalm compile -- bash -c "cd projects/yalm/src && make clean && make test USE_HIP=1 HIPARCH=gfx90a"
utils/timeit.sh yalm test -- bash -c "HIP_VISIBLE_DEVICES=2 projects/yalm/src/build/test"
HIP_VISIBLE_DEVICES=2 projects/yalm/src/build/test   # run 2 -> "All tests passed" (deterministic)
```

Build: clean, exit 0. Only the cosmetic `'-x hip' after last input file has no effect` warning (pre-existing).
Run 1: "All tests passed"
Run 2: "All tests passed" (deterministic)

AMD_LOG_LEVEL=3 confirms:
- `hipHostRegister ( ..., 2 )` -> `hipHostGetDevicePointer: Returned hipSuccess` (fix 2 active on Linux)
- `Using native code object for device: amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-`

Pass/fail counts: test_attn() CPU regression guard PASS; test_cuda_kernels() matmul/mha (attn_dot+attn_softmax+att_mix)/ffn CPU-vs-GPU epsilon 1e-4 comparisons all PASS; both runs identical.

## Re-validation after WPB/host-ptr/WARP_SIZE_MAX fixes (windows-gfx1201) 2026-06-17

Result: PASS -- windows-gfx1201 completed, validated_sha=3219ab7de814b25eb885ffe32f80b873b137a36c.

GPU: AMD Radeon RX 9070 XT (gfx1201, RDNA4, wave32). HIP_VISIBLE_DEVICES=0. ROCm 7.14.0a20260604 (TheRock, _rocm_sdk_devel, hipcc AMD clang 23.0, MSVC target).

validated_sha f10d84c was unreachable (force-pushed history); binary-equivalence carry-forward not possible. Full real-GPU revalidation performed.

Three functional fixes validated at 3219ab7:
1. matmul_wpb now runtime-derived (`max_threads_per_block / warp_size`): yields 32 on wave32 (gfx1201), so launch geometry is `<<<rows/32, 32*32=1024>>>`. Both 16 and 32 divide 4096 and 32000.
2. host_ptr_to_device now always uses `hipHostGetDevicePointer` on HIP (not `_WIN32`-gated); `hipHostRegister` passes `hipHostRegisterMapped` (flag=2). Confirmed by AMD_LOG_LEVEL=3: `hipHostRegister(..., 2)` -> `hipHostGetDevicePointer: Returned hipSuccess` for all output buffers.
3. WARP_SIZE_MAX conditionalized: `#if USE_HIP 64 #else 32`. On gfx1201 (wave32) shared arrays remain 64-wide (no OOB -- threadIdx.x in [0,31], slots 32..63 unused).

Build: `bash agent_space/build_yalm_gfx1201.sh` (unchanged script, no source edits needed). Wrapped with `utils/timeit.sh yalm compile`. Exit 0, only pre-existing deprecation and unused-argument warnings.

Test:
```
HIP_VISIBLE_DEVICES=0 utils/timeit.sh yalm test -- projects/yalm/src/build/win/test.exe   # run 1 -> "All tests passed"
HIP_VISIBLE_DEVICES=0 projects/yalm/src/build/win/test.exe                                # run 2 -> "All tests passed" (deterministic)
```

Kernels dispatched (palvirtual log): `_Z6matmulIfEvPKT_PKfiiPf`, `_Z8attn_dotPK6__halfPKfiiiiiPf`, `_Z12attn_softmaxPKfiiiPf`, `_Z7att_mixPK6__halfPKfiiiiiPf`, `_Z23fused_ffn_w1_w3_glu_actIfL14ActivationType0EEvPKT_S3_PKfiiPf`, `_Z6matmulIfEvPKT_PKfiiPf` (ffn w2). All 6 test-path kernels exercised. TheRock amdhip64_7.dll loaded from exe dir (not System32).

Pass/fail counts: test_attn() CPU regression guard PASS; test_cuda_kernels() matmul/mha (attn_dot+attn_softmax+att_mix)/ffn CPU-vs-GPU epsilon 1e-4 comparisons all PASS; both runs identical.

## Re-validation after squash (windows-gfx1101) 2026-06-19

Result: PASS -- windows-gfx1101 completed, validated_sha=a647be18771aace7c48386d08c98cc72d39bbded.

GPU: AMD Radeon PRO V710 (gfx1101, RDNA3, wave32). HIP_VISIBLE_DEVICES=1 (confirmed via hipInfo: Name=AMD Radeon PRO V710, gcnArchName=gfx1101, warpSize=32). ROCm 7.14.0a20260604 (TheRock, _rocm_sdk_devel, hipcc AMD clang 23.0, MSVC target).

Revalidate path: the delta f10d84c -> a647be18 is a squash commit collapsing the multi-commit history (311e1c3...3219ab7) to one clean upstream PR commit. The squash is tree-identical to pre-squash tip 3219ab7 (git diff 3219ab7..a647be18 = 0 lines). windows-gfx1101 was last GPU-validated at 006a0fd and only carried forward (comment-only) to f10d84c; it had NOT yet validated the functional fixes in 3219ab7 (matmul_wpb runtime-derived, host_ptr_to_device unconditional on HIP, WARP_SIZE_MAX conditionalized). Full GPU re-run required.

Build: `bash agent_space/build_yalm_gfx1101.sh` (unchanged script). gfx1101 is now on HIP_VISIBLE_DEVICES=1; AOT compilation does not need GPU access (--offload-arch=gfx1101), test run uses HIP_VISIBLE_DEVICES=1.

```
bash utils/timeit.sh yalm compile -- bash agent_space/build_yalm_gfx1101.sh
bash utils/timeit.sh yalm test   -- bash -c "HIP_VISIBLE_DEVICES=1 projects/yalm/src/build/win_gfx1101/test.exe"
HIP_VISIBLE_DEVICES=1 projects/yalm/src/build/win_gfx1101/test.exe   # run 2 -> "All tests passed"
```

Run 1: "All tests passed"
Run 2: "All tests passed" (deterministic)

Pass/fail counts: test_attn() CPU regression guard PASS; test_cuda_kernels() matmul/mha (attn_dot+attn_softmax+att_mix)/ffn CPU-vs-GPU epsilon 1e-4 comparisons all PASS; both runs identical. No HSA faults, no error output.

Three functional fixes from 3219ab7 confirmed on gfx1101 (wave32, RDNA3):
1. matmul_wpb runtime-derived (max_threads_per_block / warp_size): yields 32 on gfx1101, launch geometry <<<rows/32, 32*32=1024>>>. Both 16 and 32 divide 4096 and 32000.
2. host_ptr_to_device unconditional on HIP: hipHostGetDevicePointer used for all test output buffers; works correctly under WDDM.
3. WARP_SIZE_MAX conditionalized (#if USE_HIP 64 #else 32): on gfx1101 (wave32) shared arrays are 64-wide, threadIdx.x in [0,31], slots 32..63 unused -- no OOB.
