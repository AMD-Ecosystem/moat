# Mahout (QDP / qumat) -- ROCm/HIP port plan (linux-gfx90a lead)

## Project
- Name: mahout
- Upstream: https://github.com/apache/mahout
- Default branch: main
- Base sha (shallow clone HEAD): ac30a8c954f692f0e57e53cef185497b68b61217
- Lead platform: linux-gfx90a (MI250X, ROCm 7.2.1, HIP 7.2)

IMPORTANT framing correction: the classic JVM (Java/Scala) Apache Mahout is in
maintenance mode. `main` today is a Python project, "qumat" (a quantum-computing
abstraction over Qiskit/Cirq/Braket) plus QDP (Quantum Data Plane): a Rust
workspace under `qdp/` that GPU-accelerates classical->quantum-state data
encoding with hand-written CUDA kernels, FFI'd into Rust and exposed to Python
via PyO3/maturin with zero-copy DLPack. The portable GPU surface lives entirely
in `qdp/`, not in any JVM code or a dead JNI binding. This is a real, buildable,
GPU-testable CUDA codebase.

## Existing AMD support: Triton-only, no HIP path -> PROCEED
QDP ships TWO independent GPU backends:
1. CUDA backend (`QdpEngine(backend="cuda")`): the native, performance path.
   Rust `qdp-core` -> `qdp-kernels` (6 `.cu` files, ~20 kernels) -> the `cudarc`
   crate (a CUDA-driver/runtime-only Rust wrapper) for device alloc, streams,
   events, H2D/D2H, and kernel launch. This is the engine behind the project's
   pinned-buffer pool, dual-stream overlap pipeline, buffer reuse, and the whole
   benchmark/roadmap story. On AMD it is DEAD: `qdp-kernels/build.rs` only knows
   `nvcc`, so with no CUDA toolkit the kernels are skipped and every FFI entry is
   replaced by a stub returning error code 999; the whole `qdp-core::gpu` module
   tree is additionally `#[cfg(target_os = "linux")]` + cudarc, and cudarc fails
   to find a CUDA device on an AMD box, so tests SKIP.
2. AMD backend (`QdpEngine(backend="amd"|"triton_amd")`): a SEPARATE, pure-Python
   reimplementation of the same encoders in Triton (`qdp-python/qumat_qdp/triton_amd.py`,
   `backends/amd.py`), requiring a PyTorch-ROCm + Triton-HIP runtime. It does NOT
   touch the Rust core or the CUDA kernels.

Per PORTING_GUIDE "assess existing AMD support": this is the canonical case of
"AMD supported only via a different framework (Triton), with no HIP path -> a
ROCm/HIP port of the CUDA code still adds value." The Triton backend is a
functional Python substitute that lacks the native pipeline machinery (pinned
pool, dual-stream overlap, in-Rust DLPack ownership). Porting the native
qdp-core/qdp-kernels CUDA path to HIP gives AMD parity on the engine the project
is actually built around, and makes the single Rust core run on both vendors.

Decision: PROCEED with a correctness-first mechanical CUDA->HIP port of the
native qdp path. The kernels are simple memory-bound encoders (no CUTLASS/CuTe,
no tensor cores, no warp-specialization), so a mechanical HIP translation is the
right first step; no AMD-native rewrite is warranted.

cudarc finding (the linchpin): cudarc 0.13 is CUDA-only -- no HIP/ROCm feature
flag, no roadmap for one (features are all `cuda-1xxxx`, cublas, curand, etc.).
So the Rust host layer cannot be "feature-flagged" onto ROCm; cudarc must be
displaced on the HIP build (see Port strategy).

## Build classification: cc-crate CUDA in a Rust/Cargo workspace (NOT pure-CMake, NOT a pytorch extension)
Evidence:
- No CMake anywhere; no `find_package(Torch)`; no `torch.utils.cpp_extension` /
  `CUDAExtension`. The `torch` in `pyproject.toml [dependency-groups].dev` is only
  a test/DLPack-interop dependency, not a build backend.
- The kernels build via the Rust `cc` crate in `qdp/qdp-kernels/build.rs`:
  `cc::Build::new().cuda(true).flag("-cudart=shared").flag("-std=c++17")...compile("kernels")`,
  invoking `nvcc` (guarded by `Command::new("nvcc")`), then `cargo:rustc-link-lib=cudart`.
- The Python package is built with maturin (`uv run maturin develop --manifest-path
  qdp/qdp-python/Cargo.toml`); `qdp-python` is a PyO3 `cdylib` named `_qdp`.

Because neither PORTING_GUIDE Strategy A (pure CMake `enable_language(HIP)`) nor
Strategy B (torch hipify) matches, this port uses a project-specific variant of
Strategy A's philosophy ("only the .cu translation units see the HIP toolchain;
keep host untouched; small diff"), adapted to a Rust/cc build (below).

## Port strategy: HIP behind a USE_HIP gate in build.rs + a CUDA-runtime FFI compat shim, displacing cudarc on AMD
The diff has two layers: (A) the `.cu` kernels, and (B) the Rust host that drives
them (today via cudarc). Keep CUDA byte-for-byte the default; add a HIP path.

### A. Kernels (`qdp/qdp-kernels`) -- compile the existing `.cu` as HIP
- In `qdp-kernels/build.rs`, add a HIP branch gated on an env toggle
  (`QDP_USE_HIP=1`, mirroring the existing `QDP_NO_CUDA` / `QDP_CUDA_ARCH_LIST`
  convention) OR auto-detect `hipcc`. On the HIP branch: compile the same six
  `src/*.cu` with `hipcc` (e.g. `cc::Build::compiler("hipcc")` + `--offload-arch`
  from a new `QDP_HIP_ARCH_LIST` defaulting to `gfx90a` only when unset -- never a
  hardcoded literal, per the configurable-arch lesson), and emit
  `cargo:rustc-link-lib=amdhip64` instead of `cudart`. The CUDA branch is
  unchanged. hipcc compiles CUDA spelling directly, so `.cu` need not be renamed.
- The CUDA C++ in the kernels is almost 1:1 for HIP. Required source fixes,
  guarded by `#if defined(__HIP_PLATFORM_AMD__)` (or a tiny `kernel_compat.h`
  force-included on the HIP build), live in `amplitude.cu` only:
  - `__shfl_down_sync(0xffffffff, ...)` x2: ROCm 7.x static_asserts a 64-bit mask
    (the 32-bit literal fails to COMPILE, AutoDock-GPU lesson). Provide a compat
    full-warp mask `0xffffffffffffffffULL` on HIP.
  - `int warp_id = threadIdx.x >> 5;` (x2 in block_reduce_sum / _f32): hardcodes
    /32. On gfx90a (wave64) this mislabels warps. Replace `>> 5` with a per-arch
    warp shift (6 on `__GFX9__`, else 5) or `threadIdx.x / warpSize`. NOTE: the
    `__shared__ T shared[32]` stays correct (32 >= max warps/block at 1024
    threads on wave64 = 16), and `lane = threadIdx.x & (warpSize-1)` is already
    correct. This is a real wave64 correctness bug, not just a compile fix.
  - `cuComplex`/`cuDoubleComplex`/`make_cuComplex`/`make_cuDoubleComplex` and
    `<cuComplex.h>`: HIP provides `<hip/hip_complex.h>` with `hipDoubleComplex`
    etc. Either include the HIP complex header and alias the `cu*` names on HIP,
    or (simpler) keep `<cuComplex.h>` -- hipcc accepts CUDA complex headers via
    its CUDA-compat layer; confirm at first build and fall back to the alias if
    not. `__ldg`, `rsqrt`, `truncf`, `atomicAdd/atomicOr/atomicExch`, `isfinite`,
    `__restrict__`, `<<<>>>` all work under hipcc unchanged.
  - The FFI launch wrappers take `cudaStream_t stream`; under hipcc that maps to
    `hipStream_t`. The Rust side passes an opaque `*mut c_void`, so the ABI is
    stream-as-pointer either way; no signature change needed.
- `kernel_config.h` is plain `#define`s -- no change. `iqp.cu`/`angle.cu`/
  `phase.cu`/`basis.cu`/`validation.cu` use no warp intrinsics (only dynamic
  `extern __shared__` in iqp) and should compile under hipcc unchanged.

### B. Rust host (`qdp-core`) -- displace cudarc on the HIP build
cudarc has no ROCm backend, so the HIP build must not depend on it for device
ops. The native path uses a SMALL, well-contained slice of cudarc:
`CudaDevice::new`, `htod_sync_copy`, `dtoh_sync_copy`, `alloc_zeros`,
`device_ptr()/device_ptr_mut()`, `device.synchronize()`, plus the explicit
CUDA-runtime FFI in `qdp-core/src/gpu/cuda_ffi.rs` (~13 `cuda*` calls: HostAlloc,
FreeHost, MemGetInfo, MemcpyAsync, Memset Async, Event*, Stream*, PointerGetAttributes).
Every `cuda*` in `cuda_ffi.rs` has an exact 1:1 `hip*` equivalent
(hipHostMalloc/hipHostFree/hipMemGetInfo/hipMemcpyAsync/hipMemsetAsync/hipEvent*/
hipStream*/hipPointerGetAttributes), and HIP error codes match CUDA's numerically
for the codes used.

Two viable approaches; the porter picks after a spike (record outcome in notes.md):
- Approach B1 (preferred -- thin HIP runtime shim, minimal new code): introduce a
  small internal device abstraction so the `gpu` module no longer names cudarc
  types directly on the HIP build. Replace `cuda_ffi.rs` with a `gpu_rt` module
  that is `cuda*` (extern "C" against libcudart) on CUDA and `hip*` (extern "C"
  against libamdhip64) on HIP -- same Rust signatures, different link lib and
  symbol names, selected by `#[cfg]`/feature. Provide a tiny `DeviceBuffer`
  (raw `hipMalloc`/`hipFree` + H2D/D2H + the `device_ptr()` an FFI launch needs)
  to stand in for `cudarc::CudaSlice`, and a `Device` wrapping a HIP context +
  `synchronize()`. The `QuantumEncoder` trait's `&Arc<CudaDevice>` becomes
  `&Arc<Device>` (a type alias to cudarc's on CUDA, to the shim's on HIP). This
  keeps each encoder body nearly identical; the ~121 cudarc call-sites collapse
  onto a handful of shim methods with the same names.
- Approach B2 (smaller diff, higher risk -- cudarc-over-HIP): on Linux, HIP can be
  made to satisfy cudarc by loading libamdhip64 as the CUDA driver/runtime (HIP's
  symbols are the CUDA names under its perl-hipify-compat build, and cudarc has a
  `dynamic-linking` feature). This is unproven for cudarc specifically and the
  `cudarc` `DeviceRepr`/`CudaSlice` types assume CUDA semantics; treat as a
  fallback only if B1's surface proves larger than expected.

Recommendation: B1. It is the honest "the single Rust core runs on both vendors"
outcome, keeps the NVIDIA build untouched (cudarc stays the default), and the
porter controls the exact device-op surface. Gate all of it on a Cargo feature
`hip` (in qdp-core + qdp-kernels + qdp-python) and/or the `QDP_USE_HIP` build env,
so `cargo build` with no flags is the unchanged CUDA build.

### Python / maturin
`qdp-python` (PyO3 `_qdp`) just re-exports qdp-core; once qdp-core builds with the
`hip` feature, build the extension with the feature on:
`uv run maturin develop --manifest-path qdp/qdp-python/Cargo.toml --features hip`
(or set `QDP_USE_HIP=1`). DLPack already enumerates `kDLROCM=10` and
`GpuDeviceType::Rocm`, so the HIP path tags exported tensors as ROCm with no new
plumbing. The Python `backend="cuda"` route then drives the HIP-backed Rust core
on AMD; the separate Triton `backend="amd"` route is orthogonal and left as-is.

## CUDA surface inventory
Kernels (all in `qdp/qdp-kernels/src/`), launched from `extern "C"` wrappers:
- amplitude.cu: amplitude_encode (f64/f32), amplitude_encode_batch (f64/f32),
  l2_norm + l2_norm_batch + finalize_inv_norm (f64/f32), convert_state f64<->f32.
  Uses `__shfl_down_sync` warp reduction + block reduction (warp-size sensitive),
  `__ldg`, vectorized double2/float2 loads, `atomicAdd`, `rsqrt/rsqrtf`.
- basis.cu: basis_encode + batch (f64/f32). Simple index->one-hot writes.
- angle.cu: angle_encode + batch (f64/f32). Per-qubit cos/sin product state.
- phase.cu: phase_encode + batch (f64). Unit-modulus phase factors.
- iqp.cu: iqp_encode + batch (f64). Largest/most mathematical (Hadamard +
  diagonal ZZ phases, FWT-style); dynamic `extern __shared__`; no warp intrinsics.
- validation.cu: check_finite (f32/f64), validate_and_cast_basis_indices_f32,
  check_basis_indices_usize. `atomicExch`/`atomicOr` flag writes.
Runtime/driver surface used by Rust:
- cudarc 0.13 (driver+runtime): CudaDevice, CudaSlice, alloc_zeros, htod/dtoh sync
  copy, device_ptr(_mut), synchronize. (~121 call-sites, 16 files.)
- explicit CUDA-runtime FFI (cuda_ffi.rs): cudaHostAlloc/FreeHost (pinned pool),
  cudaMemGetInfo (OOM guard), cudaMemcpyAsync/MemsetAsync, cudaEvent* (timing +
  dual-stream deps), cudaStreamSynchronize/WaitEvent, cudaPointerGetAttributes.
- Pinned host memory: `cudaHostAlloc`/`cudaFreeHost` -> hipHostMalloc/hipHostFree.
- Streams/events: dual-stream overlap pipeline in gpu/pipeline.rs + overlap_tracker.rs.
NOT present (good): no cuBLAS/cuFFT/cuRAND/cuSPARSE/cuDNN, no Thrust/CUB, no
textures/surfaces, no CUTLASS/CuTe, no managed memory, no cooperative groups.

## Risk list
- Warp size 32-vs-64 (CONFIRMED hazards in amplitude.cu): the `0xffffffff` shfl
  mask (compile error on HIP) and the `>> 5` warp-id (wave64 correctness). Fix per
  PORTING_GUIDE warp-size class. All other kernels are warp-agnostic.
- cudarc has no ROCm backend (CONFIRMED): the central porting cost. Mitigated by
  the thin `gpu_rt`/`DeviceBuffer` shim (Approach B1); the cudarc surface actually
  used is small and uniform.
- The whole `qdp-core::gpu` tree is `#[cfg(target_os = "linux")]` AND assumes
  cudarc. Adding the `hip` feature must thread cfg correctly so: (a) CUDA build
  unchanged, (b) HIP build excludes cudarc, (c) non-Linux stub still links.
- Pinned-memory + dual-stream pipeline (pipeline.rs, buffer_pool.rs,
  overlap_tracker.rs): event/stream semantics must map exactly; hipEvent_t timing
  calls are `nodiscard` in HIP headers (amgcl lesson) -- wrap returns, do not `as _`.
- complex header: confirm hipcc accepts `<cuComplex.h>`; if not, alias to
  `<hip/hip_complex.h>` on HIP (low effort, isolated to kernel includes).
- IQP correctness: float64 Hadamard/phase math; validate numerically vs the Torch
  reference and (if any NVIDIA box is reachable) vs CUDA output, not bitwise.
- Rust toolchain: edition 2024, rust-version 1.85 -> install a current stable via
  rustup (build dep; install without asking). cargo/rustc not yet on PATH here.
- maturin/uv env: `uv sync --group dev` pulls torch (ROCm wheel must match);
  the repo's `Dockerfile.qdp-amd` is a CPU/AMD scaffold (installs clang+rust+uv,
  runs `maturin develop`) -- a useful reference but it does NOT install ROCm or
  build the kernels for HIP; do not assume it already does the port.
- atomics on coarse-grained/managed memory (cudaKDTree lesson): N/A here -- buffers
  are plain device allocations (hipMalloc), and only atomicAdd/Or/Exch are used
  (atomicAdd/CAS-family are unaffected by the gfx90a managed-memory min/max bug).

## File-by-file change list (HIP path additive; CUDA default untouched)
- qdp/qdp-kernels/build.rs: add HIP compile branch (hipcc, --offload-arch from
  QDP_HIP_ARCH_LIST default gfx90a-when-unset, link amdhip64), gated by env/feature.
- qdp/qdp-kernels/src/amplitude.cu: warp-mask + warp-shift fixes under
  `#if defined(__HIP_PLATFORM_AMD__)` (or via a new force-included kernel_compat.h).
- qdp/qdp-kernels/src/kernel_compat.h (NEW, optional): HIP complex aliases + warp
  constants, force-included on the HIP compile only.
- qdp/qdp-kernels/src/lib.rs: keep extern decls; the `qdp_no_cuda` stub cfg may
  need a sibling for the HIP build so stubs are not emitted when HIP kernels exist.
- qdp/qdp-core/Cargo.toml: add `hip` feature; make `cudarc` dep
  optional / not-required under `hip`.
- qdp/qdp-core/src/gpu/cuda_ffi.rs -> generalize to a `gpu_rt` runtime shim
  (cuda* on CUDA, hip* on HIP), same Rust signatures.
- qdp/qdp-core/src/gpu/{memory.rs,encodings/*.rs,pipeline.rs,overlap_tracker.rs,
  buffer_pool.rs,validation.rs,metrics.rs,cuda_sync.rs}: replace direct cudarc
  types with the `Device`/`DeviceBuffer` alias (alias = cudarc on CUDA, shim on
  HIP). Bodies stay nearly identical.
- qdp/qdp-python/Cargo.toml: add pass-through `hip` feature -> qdp-core/hip.
- notes.md: record exact build+test commands, the B1-vs-B2 spike outcome, ROCm
  torch wheel used, and an "## Install as a dependency" note only if needed.

## Build commands (gfx90a)
Prereqs (install without asking): rustup stable toolchain; ROCm 7.2.1 already
present; a ROCm PyTorch in the uv env (host already has torch 2.x + hip 7.2).
```
# install rust if absent
curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal && . "$HOME/.cargo/env"

# kernels + core, HIP build (gfx90a), Rust tests' compile path:
cd projects/mahout/src/qdp
QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx90a cargo build -p qdp-kernels -p qdp-core --features hip

# python extension (PyO3) for the GPU/parity pytest suite:
cd projects/mahout/src
QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx90a uv run --active maturin develop \
  --manifest-path qdp/qdp-python/Cargo.toml --features hip
```
Follower arches reuse the same commit with `QDP_HIP_ARCH_LIST=gfx1100` /
`gfx1151` and no source edit (configurable-arch rule).

## Test plan
Real GPU tests (the validation gate) -- Rust integration suite in
`qdp/qdp-core/tests/` exercises actual device kernels through the (now HIP-backed)
core; today they SKIP on AMD because `CudaDevice::new(0)` returns None. After the
port, `Device::new(0)` must return a real gfx90a device so these RUN:
- gpu_norm_f32.rs, gpu_amplitude (via gpu_api_workflow.rs), gpu_angle_encoding.rs,
  gpu_basis_encoding.rs, gpu_iqp_encoding.rs, gpu_validation.rs, gpu_dlpack.rs,
  gpu_fidelity.rs, gpu_memory_safety.rs, gpu_ptr_encoding.rs, gpu_api_workflow.rs.
- qdp-kernels/tests/{amplitude_encode.rs, angle_encode.rs}.
Run serially on the single assigned GPU (MPPI lesson): 
```
cd projects/mahout/src/qdp
QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx90a cargo test -p qdp-core -p qdp-kernels --features hip -- --test-threads=1
```
Python parity (cross-checks the native HIP path against the Torch reference and,
where present, the CUDA reference):
```
cd projects/mahout/src
uv run pytest testing -q                 # qumat + qdp python suite
uv run pytest qdp/qdp-python/tests -q     # engine/backend-routing/DLPack
```
Correctness bars (not bitwise): amplitude/angle/basis/iqp outputs match the Torch
reference within tolerance (f32 ~1e-5, f64 ~1e-12); state vectors unit-norm;
L2-norm kernel matches CPU within ULP-scaled tol; validation kernels flag
NaN/Inf/out-of-range correctly; DLPack round-trips to torch with kDLROCM. Prove
determinism with two identical runs (atomicAdd in l2_norm is the only nondeterminism
source; reduction order is fixed for a given launch config, so expect bit-identical
or sub-tol agreement).
Non-GPU regression set (must not regress): all CPU Rust tests
(preprocessing.rs, types.rs, numpy.rs, parquet_io.rs, arrow_ipc_io.rs,
tensorflow_io.rs, torch_io.rs, null_handling.rs) and the qumat (non-QDP) python
tests, which are vendor-independent.

## Inter-project deps
None. QDP's GPU stack is self-contained (no rmm/raft/Thrust/CUB/cuBLAS). Do not
set `depends_on`.

## Open questions (resolve during port; not blockers)
- B1 vs B2 for displacing cudarc: spike B1 first (thin hip runtime shim +
  DeviceBuffer). If the cudarc type surface in pipeline.rs/buffer_pool.rs is
  larger than the encoders suggest, reassess.
- Does hipcc 7.2 accept `<cuComplex.h>`/`make_cuComplex` directly, or is the
  `<hip/hip_complex.h>` alias required? (1-file build probe answers it.)
- Confirm the uv env's torch is the ROCm wheel matching ROCm 7.2 so
  `from_dlpack` parity tests run on device (host base python already has
  torch+hip 7.2; ensure the uv-managed env does too, or point pytest at the
  system interpreter).
- Cosmetic: upstream gates `gpu` modules on `target_os = "linux"`; HIP is Linux
  here, so no Windows concern for gfx1151 until that follower (Windows HIP SDK is
  out of scope for the lead).
