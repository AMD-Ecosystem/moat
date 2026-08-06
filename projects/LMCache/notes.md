# LMCache notes

Port type: finish/validate an existing upstream HIP path (Strategy B, torch
hipify). Lead platform linux-gfx90a (MI250X, CDNA2, ROCm 7.2.1). Upstream
LMCache/LMCache @ dev (base 7eb57ad). Fork AMD-Ecosystem/LMCache, branch
moat-port, HEAD e1d94420.

## What the port needed

Exactly one source change: in `setup.py` `rocm_extension()`, bump the `cxx`
flag from `-std=c++17` to `-std=c++20`.

Root cause: the BUILD_WITH_HIP=1 path compiles the plain `.cpp` sources
(pybind/mem_alloc/utils/recorders) with the `cxx` flags = `-std=c++17`, while
torch's ROCm BuildExtension compiles the `.hip` sources at `-std=c++20`. The
`.cpp` TUs include the same `<torch/all.h>` headers; on torch 2.13 those use
a C++20 `requires` constrained template (`c10/core/TensorImpl.h:2516`), so the
C++17 `.cpp` compile fails with `error: unknown type name 'requires'`. Upstream
never hit this because their ROCm CI is gfx942/gfx950 on an older torch whose
headers were C++17-clean (and their NVIDIA CI does not exercise the HIP path).

No kernel / wave64 / mem_alloc change was needed. The kernels coordinate only
via `__syncthreads()` (no warp intrinsics, atomics, or textures), so wave64 is
a no-op here. `hipHostRegister` / `hipHostGetDevicePointer` work as-is on
gfx90a for regular-pinned, NUMA-bound, and shm memory (all GPU-validated).

## Arch surface (env-driven, no source literal)

`rocm_extension()` does NOT pin an arch; torch's `_get_rocm_arch_flags()`
honors `PYTORCH_ROCM_ARCH` (verified: with `PYTORCH_ROCM_ARCH=gfx90a` the
compile gets `--offload-arch=gfx90a` only). The only `gfx942,gfx950` literals
in the repo are the ARG defaults in `docker/Dockerfile.rocm*`; they are not on
the build path used here and were left untouched (editing committed Dockerfile
arch lists would only churn the fork HEAD for followers). Followers
(gfx1100/gfx1151) build the same commit with just `PYTORCH_ROCM_ARCH=<arch>`,
no source edit.

## Build (gfx90a)

The project's own `hipify_wrapper()` runs first and rewrites `csrc/` in place
(`*.cu -> *.hip`, `pybind.cpp -> pybind_hip.cpp`, `mem_alloc.cpp ->
mem_alloc_hip.cpp`, `utils.cpp -> utils_hip.cpp`, `*.cuh -> *_hip.cuh`,
recorders -> `*_hip.*`). These match `rocm_extension()`'s `hip_sources` list
exactly; the apparent "missing .hip files" in the pristine tree are
build-generated, not a bug. Hipify is clean: 0 unmapped cuda* symbols, 14
kernel-launch rewrites. All generated files are already gitignored
(`.gitignore` `/csrc/*.hip`, `/csrc/*_hip.*`, `/csrc_hip`), so the fork commit
is just `setup.py`.

Repeatable build script lives in the fork clone at
`projects/LMCache/src/build_rocm_gfx90a.sh` (NOT committed to the fork; it is a
MOAT-side helper). It cleans the stale hipified mirror, then:

    BUILD_WITH_HIP=1 CXX=hipcc PYTORCH_ROCM_ARCH=gfx90a MAX_JOBS=16 \
      python3 setup.py build_ext --inplace

Note: `--no-build-isolation` is a pip flag, NOT a setup.py build_ext flag (the
plan listed both forms; only the pip form takes it). Build deps:
`pip install ninja "packaging>=24.2" "setuptools>=77.0.3,<81.0.0" setuptools_scm wheel`.

Builds `lmcache.c_ops` (HIP GPU ext) plus the pure-CPP `native_storage_ops`,
`lmcache_redis`, `lmcache_fs` in-place. Re-hipify on any `.cu`/`.cpp` edit
(`rm -f csrc/*.hip csrc/*_hip.* && rm -rf csrc_hip` before rebuild) -- the
Strategy-B stale-mirror gotcha.

## GPU validation (HIP_VISIBLE_DEVICES=<idle GCD>, gfx90a)

This host has 4 GCDs (rocm-smi GPU[0..3], all gfx90a MI250X). The task
suggested device 7 / 1/3/4/5/6 but only 0-3 exist here; used an idle one
(e.g. 3). Run serially on one GCD (single-GPU contention causes transient
flakes if parallelized).

Test deps installed (lightweight runtime, not full vLLM/SGLang):
`prometheus_client msgspec sortedcontainers safetensors blake3 py-cpuinfo
psutil pyyaml aiofiles aiofile aiohttp httpx pyzmq redis huggingface_hub
transformers pytest-benchmark`.

Results (all PASS; skips are env-gated, not failures):
- tests/v1/test_mem_kernels.py            56 passed   (mem_kernels.cu layout transforms, H2D/D2H, cudaHostGetDevicePointer)
- tests/v1/test_mp_mem_kernels.py         40 passed   (mp_mem_kernels.cu block transfer)
- tests/v1/test_c_ops_fallback_parity.py  \
  tests/v1/test_python_ops_fallback.py    105 passed, 1 skip  (c_ops == pure-Python ref for ALL ops, incl. encode_fast_new / decode_fast_new / decode_fast_prefsum / calculate_cdf / rotary_embedding_k_fused)
- tests/v1/test_memory_management.py      47 passed, 3 skip   (mem_alloc.cpp; 3 hugepage skips = host has no vm.nr_hugepages)
- tests/v1/test_gpu_connector.py          ~32 passed, 1 skip  (see flake note below)
- tests/v1/mp_observability/test_event_recorder.py \
  tests/v1/multiprocess/test_completion_recorder.py  17 passed (hipLaunchHostFunc host-callback path)

CacheGen arithmetic-coder determinism + round-trip (agent_space/lmcache/cachegen_determinism.py):
calculate_cdf -> encode_fast_new -> collect_bytes -> decode_fast_prefsum, 5
configs incl. 64-layer x 1024-channel x 256-token (11.2 MB). All: decode
recovers the encoded symbols byte-for-byte (round-trip exact) AND the CDF,
bytestream, lengths, and decoded output are bit-identical across two
same-seed runs (the integer-only coder is deterministic on gfx90a).

Non-GPU regression (must not regress; all PASS):
- tests/test_serde.py tests/v1/test_config.py tests/v1/test_token_database.py tests/v1/test_cache_policy.py  89 passed, 8 skip
- tests/v1/native_storage_ops/ tests/v1/storage_backend/test_fs_connector.py  330 passed
  (the pure-CPP native_storage_ops bitmap/ttl_lock/pattern_matcher are compiled by hipcc at c++17 and are unaffected)

## Gotchas / environment caveats (not gfx90a kernel issues)

- numpy vs torch.numpy() bridge: this torch (2.13.0a0, built against numpy
  1.x) raises "Numpy is not available" from tensor.numpy() under numpy >= 2.0.
  The gpu_connector tests call `.numpy()` (gpu_connectors.py:1481), so they
  need numpy < 2.0 (used 1.26.4). This is a host wheel-ABI mismatch, not a
  kernel fault.
- cupy-rocm-7-0 (14.1.0) was built against numpy 2.x and fails to import under
  numpy < 2.0 ("numpy.core.multiarray failed to import"). The two recorder
  tests import cupy, so they need numpy >= 2.0. This conflicts with the
  gpu_connector numpy<2 need -- the two test groups can't share one numpy in
  this env. Both groups were validated separately (recorders 17/17 under numpy
  2.2.6; connectors under 1.26.4). Neither is a gfx90a issue.
- test_pos_kernels.py requires `vllm` (imports vllm rotary_embedding) -- out of
  scope for a kernel port (vLLM-ROCm is heavy/engine-coupled). The pos_kernels
  fused-rope kernel is already covered by the parity test
  (test_2_compare[rotary_embedding_k_fused] PASSED), which compares c_ops vs
  the pure-Python reference directly.
- test_gpu_connector.py single-GPU flake: ~1-in-6 full-file runs, ONE
  parametrization fails an equality assert at tests/v1/utils.py:470 after a
  paged-KV async round-trip; the failing parametrization MOVES between runs
  and the flaky params pass 0/10 failures when run in isolation. This is the
  single-GPU test-harness contention/sync flake (PORTING_GUIDE line 165), not
  a deterministic kernel fault -- the underlying multi_layer_kv_transfer
  kernels pass deterministically in test_mem_kernels.py (56/56 every run).
- Hugepage register path: the host has no reserved hugepages and changing
  vm.nr_hugepages system-wide is out of bounds on this shared host. A direct
  microtest showed alloc_hugepage_pinned_ptr fails at `mmap` (MAP_HUGETLB), NOT
  at hipHostRegister -- so the register code is HIP-clean; only host config
  blocks the hugepage test. The identical hipHostRegister call on
  non-hugepage NUMA/shm memory is validated and passes.

## Inter-project MOAT deps

NONE. The c_ops extension has no build dependency on any other MOAT-ported
CUDA library (no cuBLAS/cuFFT/cuSPARSE/Thrust/CUB/CUTLASS/cuCollections/rmm/
raft). cupy-rocm-7-0 and vLLM-ROCm are pip/runtime deps, not build deps.

## Review 2026-05-31

Reviewed AMD-Ecosystem/LMCache @ moat-port (HEAD e1d94420) vs base 7eb57ad via the
/pr-review skill (local-branch mode). The diff is a single setup.py edit (9+/5-)
in rocm_extension(): the ROCm cxx flag -std=c++17 -> -std=c++20, plus comments.

No problems found. Independently verified:
- Scope: only rocm_extension()'s cxx changed. cuda_extension (setup.py:246),
  sycl_extension (setup.py:354), and _common_cpp_extensions (189/197/205) stay
  c++17 byte-identical to base; mooncake was already c++20 (132). CUDA and SYCL
  paths are untouched.
- C++20 root cause confirmed against the installed torch
  (2.13.0a0+gitb5e90ff, hip 7.2.53211): c10/core/TensorImpl.h:2516 is
  `template <typename T> requires std::is_integral_v<T> bool SetDimsTemplate(...)`,
  a C++20 requires-clause; a .cpp TU including <torch/all.h> at c++17 fails with
  "unknown type name 'requires'". Matching the .cpp TUs to the .hip TUs' c++20 is
  the minimal correct fix.
- Arch surface: no gfx literal or --offload-arch in setup.py's build path; arch
  is env-driven via PYTORCH_ROCM_ARCH. The only gfx942/gfx950 literals are the
  ARG defaults in docker/Dockerfile.rocm{,-lightweight} (off the build path);
  leaving them untouched is correct minimal-footprint (editing would churn HEAD
  and force follower revalidation).
- wave64 no-op confirmed by inspection: zero warp intrinsics
  (__shfl/__ballot/__activemask), zero atomics, zero textures/surfaces, zero
  cooperative groups in csrc. The names warp_scan (ac_enc.cu:90) and warp_copy
  (mp_mem_kernels.cu:152) are misleading but their bodies are wave-size-agnostic:
  warp_scan is a block-wide Hillis-Steele scan over blockDim.x synced by
  __syncthreads(); warp_copy is a blockDim.x-stride copy. All `32` literals are
  uint32 bit-widths (ac_enc.cu:28/46/65/70), switch(block_size) template-dispatch
  cases, or a perf cap thread_dim_x=min(elements_per_head,32) (mp_mem_kernels.cu:345)
  -- none assume warpSize, and the cap is unguarded so CUDA and ROCm match. All
  block dims are min(...,128/512) or template tile sizes with stride loops.
- Hugepage path: mem_alloc.cpp alloc_hugepage_pinned_ptr (49) calls _mmap_anon
  (51) which mmap(MAP_HUGETLB|MAP_HUGE_2MB) (24-26) and throws on failure (28)
  BEFORE cudaHostRegister (53). On a host with 0 reserved hugepages it fails at
  mmap, not at hipHostRegister, so the register path is HIP-clean; the same
  register call on NUMA (149) and shm (209) memory is reached and was validated.
- Hygiene: title [ROCm] ... 62 chars (<=72); body has root cause + Test Plan +
  Claude disclosure; no Co-Authored-By/noreply, no ghstack, no signed-off, no
  AMD-internal GitHub account (fork is AMD-Ecosystem/LMCache); ASCII, no em-dash.
- Commit is setup.py-only: hipify mirrors (/csrc/*.hip, /csrc/*_hip.*, /csrc_hip)
  are gitignored (.gitignore 119-123) and none are tracked; working tree carries
  only the uncommitted MOAT helper build_rocm_gfx90a.sh (intentionally not in the
  fork).

Verdict: review-passed. The GPU test suite (parity 105, mem-kernels 56,
mp-mem 40, memory-mgmt 47/3-skip, recorders 17, CacheGen 5/5 bit-exact) is the
load-bearing evidence; the porter ran it but the validator stage re-runs it on
real GPU next, which is the gate. Safe to proceed to GPU validation.

## Validation 2026-05-31

Platform: linux-gfx90a (AMD Instinct MI250X / MI250, gfx90a, ROCm 7.2.1 / hip 7.2.53211)
Validated SHA: e1d94420d5d96b4946d26704b5b0350b88fb11d6
GCD used: 2 (HIP_VISIBLE_DEVICES=2; GCDs 1 and 3 were at 100% GPU utilization at run time)

Build command (incremental, near-no-op):

    cd /var/lib/jenkins/moat/projects/LMCache/src
    BUILD_WITH_HIP=1 CXX=hipcc PYTORCH_ROCM_ARCH=gfx90a python3 setup.py build_ext --inplace

Result: c_ops.cpython-312-x86_64-linux-gnu.so relinked cleanly; all four .so outputs confirmed present.

GPU test results (all vs documented bar):

- tests/v1/test_mem_kernels.py            56 passed          [BAR: 56] PASS
- tests/v1/test_mp_mem_kernels.py         40 passed          [BAR: 40] PASS
- tests/v1/test_c_ops_fallback_parity.py +
  tests/v1/test_python_ops_fallback.py    105 passed, 1 skip [BAR: 105p/1s] PASS
- tests/v1/test_memory_management.py      47 passed, 3 skip  [BAR: 47p/3s] PASS
- tests/v1/test_gpu_connector.py          32 passed, 1 skip  [BAR: ~32p/1s] PASS (no flake this run)
- event_recorder + completion_recorder    17 passed          [BAR: 17] PASS
  (ran under numpy 2.4.6 for cupy-rocm compat; restored to 1.26.4 after)

CacheGen determinism (agent_space/lmcache/cachegen_determinism.py):

    5/5 configs PASS (nl=8/16/64/2/32; max 11.2 MB)
    All: roundtrip_exact=True, det(cdf=True, buf=True, len=True, dec=True)

Non-GPU regression:

- tests/test_serde.py tests/v1/test_config.py tests/v1/test_token_database.py
  tests/v1/test_cache_policy.py                89 passed, 8 skip [BAR: 89p/8s] PASS
- tests/v1/native_storage_ops/ tests/v1/storage_backend/test_fs_connector.py
                                              330 passed          [BAR: 330] PASS

All documented pass counts met or exceeded; all documented skips reproduced exactly. No regressions.
Transition: review-passed -> completed (validated_sha = e1d94420; followers unblocked to port-ready).

## Validation 2026-05-31 (gfx1100, ROCm 7.2.1)

Platform: linux-gfx1100 (AMD Radeon Pro W7800 48GB, gfx1100, RDNA3, wave32, ROCm 7.2.1 / hip 7.2.53211)
Validated SHA: e1d94420d5d96b4946d26704b5b0350b88fb11d6 (fork untouched; zero source delta vs lead)
GPU used: HIP_VISIBLE_DEVICES=0

Build command (Strategy B; arch env-driven, no source edit):

    cd /var/lib/jenkins/moat/projects/LMCache/src
    BUILD_WITH_HIP=1 CXX=hipcc PYTORCH_ROCM_ARCH=gfx1100 MAX_JOBS=16 \
      python3 setup.py build_ext --inplace

Result: clean; hipify 0 unmapped symbols, 14 kernel-launch rewrites (same as gfx90a); all 4 .so files built.

gfx1100 code-object evidence (roc-obj-ls on lmcache/c_ops.cpython-312-x86_64-linux-gnu.so):
6 code objects, all hipv4-amdgcn-amd-amdhsa--gfx1100; no gfx90a present.
Compile flags visible in output: --offload-arch=gfx1100 (steps 5-10 of 11).

GPU test results vs gfx90a bar:

- tests/v1/test_mem_kernels.py              56 passed          [BAR: 56] PASS
- tests/v1/test_mp_mem_kernels.py           40 passed          [BAR: 40] PASS
- tests/v1/test_c_ops_fallback_parity.py +
  tests/v1/test_python_ops_fallback.py      105 passed, 1 skip [BAR: 105p/1s] PASS
  (c_ops == pure-Python reference for ALL ops on wave32: encode_fast_new, decode_fast_new,
   decode_fast_prefsum, calculate_cdf, rotary_embedding_k_fused, all kv-transfer variants)

Determinism: test_mem_kernels.py re-run -> 56 passed (identical to run 1). Kernel layout
transforms + H2D/D2H reproduce bit-for-bit.

Non-GPU regression:

- tests/test_serde.py tests/v1/test_config.py tests/v1/test_token_database.py
  tests/v1/test_cache_policy.py             89 passed, 8 skip  [BAR: 89p/8s] PASS
- tests/v1/native_storage_ops/ tests/v1/storage_backend/test_fs_connector.py
                                            330 passed          [BAR: 330] PASS

No fork interaction: moat-port head e1d94420 untouched; no CI workflow added.
Transition: port-ready -> completed (validated_sha = e1d94420).

## Windows gfx1151 attempt 2026-05-30 -- BLOCKED (never Windows-ported; non-ROCm POSIX host porting out of scope)

Platform: gfx1151 APU, Windows, TheRock ROCm (gsplat venv torch 2.12+rocm). The ROCm/HIP
GPU surface is fine: setup.py's BUILD_WITH_HIP hipify runs cleanly ("Successfully
preprocessed all matching files", 0 unsupported CUDA calls), and torch builds the c_ops
HIP extension path. The blocker is NOT ROCm and NOT gfx1151 -- LMCache has NEVER been
ported to Windows (even for CUDA): grep of csrc/ finds ZERO _WIN32/_MSC_VER guards. Its
host C++ is pervasively POSIX:
- csrc/storage_manager/bitmap.cpp used GCC builtins __builtin_popcount/__builtin_ctz (no
  MSVC equiv) -- fixed with an _MSC_VER shim (see windows-partial-fixes.patch).
- csrc/storage_backends/redis/* (event_notifier.h) needs <unistd.h>/eventfd -- POSIX sockets.
- **csrc/mem_alloc.cpp (the c_ops host allocator, the layer memory_management.py calls via
  alloc_pinned_numa_ptr / alloc_hugepage_pinned_ptr / alloc_shm_pinned_ptr) needs
  <sys/mman.h> + <unistd.h>** (mmap, hugepages, shm) -- no Windows equivalent without a
  VirtualAlloc/large-page/Win32-shared-memory port.

So GPU validation on Windows requires porting LMCache's POSIX host layer (NUMA/hugepage/shm
pinned allocators + redis/fs storage backends) to Win32 -- a substantial Windows-CUDA
portability effort that is SEPARATE from the ROCm/HIP port and out of scope here. The ROCm
HIP kernels themselves (mem_kernels.hip) are unaffected (they hipify). gfx90a/gfx1100 (Linux)
pass. Blocked; revisit if/when LMCache gains a Windows host-allocator/networking port upstream.

## Validation 2026-06-04 (windows-gfx1101) -- BLOCKED (same POSIX host-layer blocker as gfx1151)

Platform: windows-gfx1101, AMD Radeon PRO V710 (RDNA3, gfx1101), TheRock ROCm 7.14.0a20260604,
torch 2.9.1+rocm7.14.0a20260604. HIP_VISIBLE_DEVICES=0.

Build attempt:

    VENV=B:/develop/TheRock/external-builds/pytorch/.venv
    export PATH="$VENV/Lib/site-packages/_rocm_sdk_devel/bin:..."
    export HIP_VISIBLE_DEVICES=0
    export BUILD_WITH_HIP=1 CXX=hipcc PYTORCH_ROCM_ARCH=gfx1101 DISTUTILS_USE_SDK=1
    cd B:/develop/moat/projects/LMCache/src
    python setup.py build_ext --inplace

HIP hipify stage: "Successfully preprocessed all matching files", 0 unsupported CUDA
calls, 0 replaced kernel launches -- ROCm/HIP GPU surface is clean on Windows too.

Build failure: identical to gfx1151. The first extension built is `native_storage_ops`:
`csrc/storage_manager/bitmap.cpp` references `__builtin_popcount`/`__builtin_ctz` (GCC
builtins, not available in MSVC). The `windows-partial-fixes.patch` has a fix for that,
but that is not on the fork commit. Even if applied, the c_ops extension would fail next:
`csrc/mem_alloc.cpp` includes `<sys/mman.h>`, `<sys/syscall.h>`, `<linux/mman.h>`,
`<linux/mempolicy.h>` -- Linux kernel headers, unconditionally (zero `_WIN32`/`_MSC_VER`
guards anywhere in csrc/). No Windows equivalent for the POSIX+Linux mmap/NUMA/hugepage/
shm pinned-allocator stack without a full Win32 rewrite (VirtualAlloc + large pages +
Win32 shared memory). This is LMCache's total lack of a Windows host port, not a ROCm
defect.

Exact first build error:

    B:\...\csrc\storage_manager\bitmap.cpp(62): error C3861: '__builtin_popcount': identifier not found
    B:\...\csrc\storage_manager\bitmap.cpp(88): error C3861: '__builtin_ctz': identifier not found

Blocker: same as gfx1151 -- not ROCm, not gfx1101. The GPU kernels hipify cleanly;
the host allocator requires porting LMCache's POSIX layer to Win32, which is out of
scope for the ROCm port. Blocked.

## Upstream-PR disposition: VALIDATED + a contributable build-rot fix (delta check 2026-06-04)

An earlier README-sweep note here said "superseded, DO NOT open a PR -- upstream already supports ROCm." Retracted; it was the most wrong of the three sweep REDs. Upstream does have mature ROCm scaffolding (merged #3070/#3092/#3101/#3395 gfx950-MI350X/#3079), but the delta check shows real, contributable value:

- CONTRIBUTABLE BUILD-ROT FIX: the fork's only diff is setup.py rocm_extension() `-std=c++17` -> `-std=c++20`. Without it, BUILD_WITH_HIP=1 fails to compile the plain .cpp TUs on CURRENT PyTorch: torch's ROCm BuildExtension compiles the .hip sources at c++20, and the .cpp sources include the same <torch/all.h>, which now uses a c++20 `requires`-constrained template (c10/core/TensorImpl.h:2516). Upstream's ROCm CI uses older torch (c++17-clean headers) + gfx942/gfx950, and NVIDIA CI never exercises the HIP path, so they never hit it. This fixes the HIP build for everyone on modern torch, ALL arches -- a genuine bugfix PR, stronger than CTranslate2's arch-list line.
- ARCH COVERAGE + FIRST GPU VALIDATION: upstream targets gfx942/gfx950 (CDNA3/CDNA4) only, with build-only ROCm CI. We GPU-validated the full c_ops suite (parity 105, mem-kernels 56, mp-mem 40, memory-mgmt 47, recorders 17, CacheGen 5/5 bit-exact) on gfx90a (CDNA2 wave64) AND gfx1100 (RDNA3 wave32) -- first evidence the HIP path runs correctly on CDNA2 + RDNA3. Wave-agnostic (syncthreads-only; warp_scan/warp_copy are misnamed block-wide ops). gfx1151 blocked only by LMCache's total lack of a Windows host port (POSIX mmap/NUMA/hugepage/shm) -- external, not ROCm.

Outcome = validated (2 arch). The c++20 fix IS PR-worthy (real build break on current torch); flagged for a decision rather than auto-skipped. See [[moat-no-duplicate-amd-ports]].
