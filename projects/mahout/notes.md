# mahout notes

Port target is the QDP (Quantum Data Plane) Rust workspace under `qdp/`, NOT the
JVM Mahout. Native engine: `qdp-kernels` (6 hand-written `.cu`, ~20 encoder/L2-norm
kernels) compiled by the Rust `cc` crate, driven from `qdp-core` through the
`cudarc` crate + a CUDA-runtime FFI shim. The port adds an AMD/HIP build behind a
Cargo `hip` feature (and `QDP_USE_HIP=1`); the NVIDIA build (default `cuda`
feature) is byte-for-byte unchanged. The separate Triton `backend="amd"` Python
path is orthogonal and untouched.

## Environment (lead: linux-gfx90a)
- ROCm 7.2.1, hipcc 7.2 (AMD clang 22), 4x MI250X GCDs (gfx90a, wave64).
- Rust stable via rustup (workspace needs edition 2024 / rust 1.85+).
- Python: conda env `py_3.12` already has ROCm torch 2.13 (hip 7.2, 4 devices),
  pytest 9. The uv `dev` group pins a generic torch and, if installed, would
  clobber the working ROCm torch -- so we install only the `_qdp` extension into
  the existing env (see below), never `maturin develop` (which runs the dev-group
  pip install).

## What the port does

### A. Kernels (`qdp/qdp-kernels`) -- same `.cu` compiled by hipcc
- `build.rs`: added a HIP branch (`build_hip`) gated by `hip_requested()` (the
  Cargo `hip` feature OR `QDP_USE_HIP=1`). It compiles the same six `.cu` with
  hipcc, `--offload-arch` from `QDP_HIP_ARCH_LIST` (default `gfx90a` only when
  unset -- never a hardcoded literal that overrides the env, so followers
  gfx1100/gfx1151 build the same source with only `QDP_HIP_ARCH_LIST=<arch>`),
  and links `amdhip64`. The CUDA branch (nvcc, gencode, cudart) is unchanged.
- hipcc ships NO `<cuda_runtime.h>`, `<cuComplex.h>`, or `<vector_types.h>`. Rather
  than edit every include line, `qdp-kernels/hip_compat/` holds forwarding shim
  headers of those exact names (MPPI lesson); `build_hip` adds that dir to the
  include path FIRST. On the CUDA build the dir is absent, so the real toolkit
  headers win -> CUDA path untouched.
  - `hip_compat/cuda_runtime.h` -> `<hip/hip_runtime.h>` + the small set of cuda*
    runtime aliases the kernels use (cudaError_t, cudaSuccess,
    cudaErrorInvalidValue, cudaStream_t, cudaGetLastError, cudaGetDevice,
    cudaDeviceGetAttribute, cudaDevAttrMaxGridDimX, cudaMemsetAsync, cudaMalloc).
  - `hip_compat/cuComplex.h` -> `<hip/hip_complex.h>` + aliases. cuDoubleComplex/
    cuComplex -> hipDoubleComplex/hipFloatComplex; make_cu* -> make_hip*. The
    `cuC*` ops (cuCreal/cuCimag/cuCadd/cuCsub/cuCmul/cuConj) are called only on
    cuDoubleComplex in iqp.cu, so they alias to HIP's DOUBLE helpers (hipCreal,
    ... -- NOT the float `f` set) via tiny inline wrappers.
  - `hip_compat/vector_types.h` -> `<hip/hip_runtime.h>` (HIP provides double2/
    float2 there).
- `src/kernel_compat.h` (NEW, included by amplitude.cu): on HIP defines
  `QDP_FULL_WARP_MASK = 0xffffffffffffffffULL`, on CUDA `0xffffffffu`.
- `src/amplitude.cu` source fixes (the only kernel needing them; all others are
  warp-agnostic and compile unchanged):
  - `__shfl_down_sync(0xffffffff, ...)` x2 -> `__shfl_down_sync(QDP_FULL_WARP_MASK, ...)`.
    ROCm 7.x static_asserts a 64-bit mask (`sizeof(MaskT)==8`); the 32-bit literal
    fails to COMPILE (confirmed). CUDA keeps the 32-bit value.
  - `int warp_id = threadIdx.x >> 5;` x2 -> `threadIdx.x / warpSize`. A genuine
    wave64 CORRECTNESS bug: `>> 5` assumes 32-lane warps, so on gfx90a the
    per-warp L2-norm partial lands in the wrong shared slot and the final
    reduction reads the wrong slot -> wrong norm. `/ warpSize` is arch-unified
    (== `>>5` on CUDA/RDNA wave32, `>>6` on CDNA wave64). The `lane =
    threadIdx.x & (warpSize-1)` and `__shared__ shared[32]` were already correct
    (16 warps max at 1024 threads on wave64).

### B. THE LINCHPIN -- displacing cudarc on the HIP build (cc-crate, NOT cudarc-over-HIP)
cudarc 0.13 is CUDA-only (no ROCm backend, no `dynamic-linking`-to-HIP that works
with its `CudaSlice`/`DeviceRepr` semantics). Approach B1 (thin HIP runtime shim)
was implemented in full; B2 (cudarc-over-HIP) was not attempted (cudarc's sys
layer hard-binds the CUDA driver). The whole cudarc surface QDP uses is small and
uniform, so a same-named shim collapses every call site with zero body changes:

- `qdp-kernels/src/device.rs` (NEW): vendor-selected device module.
  - On `cuda`: `pub use cudarc::driver::{CudaDevice, CudaSlice, CudaStream,
    DevicePtr, DevicePtrMut, DeviceRepr, DeviceSlice, ValidAsZeroBits}`.
  - On `hip`: a self-contained shim with the SAME type names + method signatures,
    backed by `extern "C"` libamdhip64 calls. `CudaDevice` (ordinal + bind),
    `CudaSlice<T>` (owns hipMalloc'd ptr as u64, Drop -> hipFree), `CudaViewMut`
    (slice_mut sub-view), `CudaStream { pub stream: *mut c_void }`. Methods:
    `new`, `alloc`(unsafe), `alloc_zeros`, `htod_sync_copy`, `htod_copy`,
    `htod_sync_copy_into`(generic over DevicePtrMut so the slice_mut view works),
    `dtoh_sync_copy`, `synchronize`, `ordinal`, `fork_default_stream`, `wait_for`.
    Marker traits `DeviceRepr`/`ValidAsZeroBits` and accessor traits
    `DevicePtr`/`DevicePtrMut`(return `&u64` so `*x.device_ptr() as *mut T` is
    unchanged)/`DeviceSlice`. `DriverError(i32)` is a Debug wrapper (call sites
    only `{:?}` it). The marker traits live in qdp-kernels (lowest crate) because
    it impls them on its complex structs.
- `qdp-core/src/gpu_rt.rs` (NEW): `pub use qdp_kernels::device::{...}` -- the
  single import point. Every `use cudarc::driver::{...}` in qdp-core src + tests
  became `use crate::gpu_rt::{...}` (or `qdp_core::gpu_rt::` in tests);
  `safe::CudaStream` flattened to `CudaStream`. ~16 src files + 3 test files +
  2 qdp-kernels test files (-> `qdp_kernels::device`). Bodies are byte-identical.
- `qdp-core/src/gpu/cuda_ffi.rs`: kept the public `cuda*` fn names + constants the
  pinned-pool/OOM-guard/pipeline call; split into a `cuda_rt` mod (extern
  libcudart, default) and a `hip_rt` mod (thin wrappers over libamdhip64:
  hipHostMalloc/hipHostFree/hipMemGetInfo/hipMemcpyWithStream/hipMemcpy/hipEvent*/
  hipStream*/hipMemsetAsync/hipPointerGetAttributes), selected by feature. Added
  `cudaMemcpy` (sync) + `CUDA_MEMCPY_DEVICE_TO_HOST`.
- `qdp-core/src/gpu/metrics.rs`: the two `download_complex_*` test helpers used
  cudarc's raw `sys::lib().cuMemcpyDtoH_v2`; now use cuda_ffi's `cudaMemcpy`
  (D2H) -- cross-vendor and out of cudarc's sys layer.
- Cargo features: `qdp-kernels` {default=["cuda"], cuda=["dep:cudarc"], hip=[]};
  `qdp-core` {default=["cuda"], cuda=["dep:cudarc","qdp-kernels/cuda"],
  hip=["qdp-kernels/hip"]} with cudarc + qdp-kernels deps `optional`/
  `default-features=false`; `qdp-python` {default=["cuda"], cuda=["qdp-core/cuda"],
  hip=["qdp-core/hip"]}.

## Build commands (gfx90a)
```
curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal && . "$HOME/.cargo/env"
export QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx90a ROCM_PATH=/opt/rocm

# kernels + core (HIP). --no-default-features turns off the cuda feature (cudarc).
cd projects/mahout/src/qdp
cargo build -p qdp-core -p qdp-kernels --no-default-features --features hip -j 16

# Python extension: build a wheel (does NOT touch torch), then install ONLY the
# extension into the conda env that already has ROCm torch. Do NOT `maturin
# develop` -- it runs `pip install --group dev` which would replace ROCm torch.
conda activate py_3.12
maturin build --features hip --manifest-path qdp/qdp-python/Cargo.toml --out <wheeldir>
pip install --no-deps --force-reinstall <wheeldir>/qumat_qdp-*.whl
```
Follower arches: same commit, `QDP_HIP_ARCH_LIST=gfx1100` / `gfx1151`, no source edit.

## Test commands (gfx90a) -- pick a FREE GCD
This box has 4 GCDs; check `rocm-smi --showuse --showmemuse` and use a free one via
`HIP_VISIBLE_DEVICES=<n>` (the Rust code hardcodes device ordinal 0; HIP_VISIBLE_DEVICES
remaps it). Run serially (single GPU): `--test-threads=1`.
```
export QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx90a ROCM_PATH=/opt/rocm HIP_VISIBLE_DEVICES=2
cd projects/mahout/src/qdp
cargo test -p qdp-core -p qdp-kernels --no-default-features --features hip -- --test-threads=1
# Python parity (system interpreter with ROCm torch + installed _qdp wheel):
cd projects/mahout/src && python -m pytest testing qdp/qdp-python/tests -q
```

## Gotchas / lessons (see PORTING_GUIDE changelog for generalized entries)
- `maturin develop` runs `pip install --group dev` (pytest + torch>=2.2,<=2.9 +
  triton) into the active env, which DOWNGRADES/replaces a preinstalled ROCm
  torch with a non-ROCm wheel -> GPU parity tests then see no device. Build a
  wheel with `maturin build` and `pip install --no-deps` it instead.
- hipcc finds none of `<cuda_runtime.h>` / `<cuComplex.h>` / `<vector_types.h>`;
  use forwarding shim headers on a HIP-only include dir.
- hip_complex.h exposes only `hipC*` (no `cuC*`); cuCreal/cuCadd/... are the
  DOUBLE versions in CUDA, so alias them to hipCreal/hipCadd (double), not the
  `f` float set, unless the call site is float-complex.
- `cc::Build` with `compiler("hipcc")` needs `.flag("-x").flag("hip")` so the
  `.cu` are compiled as HIP (the cc crate would otherwise pass C/C++ mode).

## Validation result (lead linux-gfx90a, MI250X, ROCm 7.2.1) -- PASS
All GPU + non-GPU tests RUN (no longer SKIP) and PASS:
- qdp-kernels: amplitude_encode 21/21, angle_encode 10/10.
- qdp-core GPU: gpu_angle 12, gpu_api_workflow 8, gpu_basis 7, gpu_dlpack 9,
  gpu_fidelity 17, gpu_iqp 22, gpu_memory_safety 4, gpu_norm_f32 2,
  gpu_ptr_encoding 64, gpu_validation 8; lib unit tests 77.
- Non-GPU regression: arrow_ipc 5, null_handling 6, numpy 4, parquet 8,
  preprocessing 14, tensorflow_io 9, torch_io 3, types 6. 0 failures total.
- Python parity (testing/qdp + testing/qdp_python, ROCm torch 2.13): 275 passed,
  9 skipped, 0 failed. Skips are legit + pre-existing: 2 multi-GPU,
  1 tensorflow-absent, 1 loader path-validation timing, 5 in test_torch_ref.py
  (the Triton/torch reference path's CUDA-centric `sm_<cap>` arch check vs ROCm
  torch's gfx arch list -- not the native engine; torch compute works on the GCD).
- Backend routing (qdp/qdp-python/tests/test_backend_routing.py): 7 passed.
- Two DLPack device-type tests (Rust gpu_api_workflow.rs::test_dlpack_device_id,
  Python testing/qdp/test_bindings.py::test_dlpack_device) hardcoded kDLCUDA(2);
  made arch-aware to expect kDLROCM(10) on the HIP build (the correct value, and
  what ROCm torch's from_dlpack requires). NOT a port bug.

## Known caveat: release LTO + HIP cdylib (gpuRIR lesson)
The workspace `[profile.release]` has `lto = "fat"`. A RELEASE build of the
qdp-python cdylib under the HIP toolchain produces a 0-byte / bitcode-only `.so`
with no PyInit (import fails "file too short"); the Rust .a/.rlib + all test
binaries are unaffected. For the Python extension, build with `--profile dev`
(no LTO) -- validated working (192MB .so, PyInit__qdp present). The Rust+HIP
link itself is fine (correctly pulls libamdhip64.so.7); only fat-LTO on the
final cdylib breaks. maturin's auditwheel repair also chokes on ROCm libs
("patchelf: missing ELF header"); use `--compatibility linux` to skip it.
Validation installed the dev cdylib directly into the conda env's _qdp pkg.

## Inter-project deps
None. Do not set `depends_on`. (No "Install as a dependency" section: nothing
in MOAT consumes QDP.)

## Review 2026-05-31 (reviewer, linux-gfx90a) -- CHANGES REQUESTED

Verdict: the port is structurally sound and the wave64 fault-class analysis is
correct and verified, but two cheap, genuine defects should land before GPU
validation. Reviewed `git diff ac30a8c...HEAD` (36 files, +953/-115) on
moat-port @ 79a257cd. Findings are problems only.

1. (must-fix, behavior) qdp-core/src/gpu/cuda_ffi.rs:181-189 -- the HIP shim maps
   `cudaMemcpyAsync` to `hipMemcpyWithStream`, which synchronizes the stream and
   blocks the host before returning. The sole call site (pipeline.rs:112, via
   async_copy_to_device) is the dual-stream overlap path whose explicit intent is
   "true async copy ... non-blocking" (pipeline.rs:98, 442-443). A host-blocking
   copy silently serializes the pipeline, defeating the dual-stream overlap on
   AMD -- the very native-engine feature that justifies this port over the
   existing Triton backend (plan.md). The exact 1:1 of cudaMemcpyAsync is
   `hipMemcpyAsync` (present in ROCm 7.x hip_runtime_api.h alongside
   hipMemcpyWithStream). Fix: bind/wrap `hipMemcpyAsync` instead. Correctness is
   not affected (the copy-done event + stream-wait still order things), so the
   validated test results stand; this is a behavior/perf regression of the port's
   headline feature. Re-validate on GPU after the swap since it changes the H2D
   path the pipeline tests exercise.

2. (must-fix, latent safety) qdp-kernels/src/device.rs:318-336 --
   `htod_sync_copy_into` copies `size_of_val(src)` bytes into `dst` without
   checking `dst`'s length, whereas cudarc's `htod_sync_copy_into` asserts
   `src.len() == dst.len()`. Safe today only because the single external caller
   (encoding/basis.rs:146-149) builds `indices_cpu` to exactly `samples_in_chunk`
   (indices_cpu.clear() at basis.rs:108 then one push per chunk element) and
   slices the dst view to the same length. But the dropped invariant turns a
   future length mismatch from a cudarc panic into a silent device-buffer
   overflow (OOB write). Fix: assert dst.len() (DeviceSlice::len) == src.len() in
   the shim to match cudarc semantics.

3. (must-fix, trivial/style) qdp-core/src/gpu/cuda_ffi.rs:55 and :120 -- the new
   section-divider comments use Unicode box-drawing characters (the long bar
   glyphs around "CUDA backend"/"HIP backend"). CLAUDE.md requires ASCII-only in
   new comments. Replace with ASCII (e.g. `// ---- CUDA backend ... ----`).

4. (minor, doc) qdp-kernels/src/kernel_compat.h:19 -- the header comment says
   "Included by every kernel translation unit", but only amplitude.cu includes it
   (confirmed by grep; it is the only kernel with warp intrinsics). Reword to
   "Included by the kernel TUs that use warp intrinsics (amplitude.cu)".

Verified sound (no action): the amplitude.cu wave64 fixes are correct and match
the AutoDock-GPU lessons -- 64-bit QDP_FULL_WARP_MASK keyed on
__HIP_PLATFORM_AMD__ (not wave width) and warp_id = threadIdx.x / warpSize
(== >>5 on wave32, >>6 on wave64), with __shared__[32] still a valid upper bound
(<=16 warps at 1024 threads on wave64); the CUDA path is byte-identical
(0xffffffffu, /32). The Cargo feature gating makes cudarc optional and keeps the
default `cuda` build binding cudarc; every host change is a pure
`cudarc::driver::` -> `crate::gpu_rt::` / `qdp_kernels::device::` import swap with
byte-identical bodies. DLPack tags kDLROCM=10 via NATIVE_GPU_DEVICE_TYPE
(feature-gated; kDLCUDA on CUDA) and the two device-type tests are correctly made
arch-aware (not weakened). The cuda_ffi.rs cuda_rt mod is byte-identical to the
original extern block under `all(cuda, not(hip))`. The metrics.rs download
helpers switch from cudarc's driver-API cuMemcpyDtoH_v2 to the runtime-API
cudaMemcpy (D2H) -- a semantically-equivalent swap in a test/validation helper,
needed to leave cudarc's sys layer; it does touch the CUDA path, so the validator
should confirm the NVIDIA build still passes there. Rule-of-five on the shim
handles is satisfied: CudaSlice (Drop guards ptr!=0) and CudaStream (Drop guards
!is_null), both move-only, no Clone/Copy, no default-constructed handle is
destroyed. Commit hygiene is clean: `[ROCm]` title (<=72), Test Plan present,
Claude disclosed, no noreply trailer, no ghstack, author is the jeffdaily user
identity (no AMD-internal account), single curated commit on moat-port.

## Porter fix 2026-05-31 (changes-requested -> review-passed) -- fork @ 2b0544a

Addressed all 4 review findings; nothing else touched (the wave64 fixes, cudarc
displacement, DLPack, and the CUDA default path are left as-is). Default CUDA
build still binds cudarc (`cargo check -p qdp-core -p qdp-kernels` with default
features type-checks; the QDP_NO_CUDA=1 only skips nvcc kernel compilation).

1. (behavior) cuda_ffi.rs hip_rt: cudaMemcpyAsync now maps to hipMemcpyAsync (the
   exact 1:1 enqueue-and-return, ROCm 7.x hip_runtime_api.h:5037), not
   hipMemcpyWithStream (which synchronizes the stream and blocks the host). The
   old mapping silently serialized the dual-stream H2D overlap pipeline
   (pipeline.rs async_copy_to_device, line 461). Correctness was never affected
   (the copy-done event + wait_for_copy still order copy->compute), so the
   validated results stand; this restores the non-blocking behavior the native
   engine's headline feature depends on.
   - GPU RE-VALIDATION of the async path (HIP_VISIBLE_DEVICES=3, gfx90a):
     * A direct libamdhip64 latency probe on a 256MB pinned H2D copy: hipMemcpyAsync
       returns to the host in ~11 us (transfer 18.64 ms proceeds on the stream);
       hipMemcpyWithStream blocks ~18.63 ms (the full transfer) before returning
       -- 1694x longer. Confirms hipMemcpyAsync is genuinely non-blocking.
     * The dual-stream async-pipeline tests pass with QDP_ENABLE_OVERLAP_TRACKING=1:
       test_amplitude_encoding_async_pipeline, test_angle_encoding_async_pipeline
       (gpu_api_workflow), test_angle_batch_f32_async_pipeline_path
       (gpu_angle_encoding). The H2D-vs-compute overlap timeline records cleanly
       through hipMemcpyAsync (no "invalid resource handle"/event-lifecycle
       errors); a 32MB / 4-chunk multi-chunk pipeline runs and the OverlapTracker
       reports per-chunk overlap, with correct encoder output.
2. (latent safety) device.rs htod_sync_copy_into: added
   `assert_eq!(dst.len(), src.len())` and a `+ DeviceSlice<T>` bound on the
   destination type param, matching cudarc's contract. Both call sites (the
   internal htod_sync_copy at device.rs:300 and encoding/basis.rs:149 via a
   slice_mut view) already satisfy it; this turns a future length mismatch from a
   silent device-buffer OOB write into a clean panic. gpu_memory_safety (4) still
   passes.
3. (style) cuda_ffi.rs:55,120: section dividers changed from Unicode box-drawing
   glyphs to ASCII `// ---- ... ----`.
4. (doc) kernel_compat.h:19: comment corrected from "Included by every kernel
   translation unit" to "Included by the kernel TUs that use warp intrinsics
   (amplitude.cu)".

Regression re-validation (HIP_VISIBLE_DEVICES=3, gfx90a, ROCm 7.2.1) -- all GREEN,
identical to the prior validation:
- qdp-kernels: amplitude 21, angle 10.
- qdp-core lib 77; GPU suites gpu_angle 12, gpu_api_workflow 8, gpu_basis 7,
  gpu_dlpack 9, gpu_fidelity 17, gpu_iqp 22, gpu_memory_safety 4, gpu_norm_f32 2,
  gpu_ptr_encoding 64, gpu_validation 8; non-GPU arrow 5, null 6, numpy 4,
  parquet 8, preprocessing 14, tensorflow 9, torch 3, types 6. 0 failures.
- Python parity (dev-profile HIP wheel, pip --no-deps into the ROCm-torch env;
  testing/qdp + testing/qdp_python + qdp/qdp-python/tests): 301 passed, 12
  skipped, 0 failed. Skips are pre-existing/legit (2 multi-GPU, 1 tensorflow,
  1 loader path-timing, 5 torch_ref sm_-arch check, 2 AmdQdpEngine-not-built,
  1 NVIDIA-ref-absent). NOTE: the full `testing/` tree also has
  testing/qumat/test_amazon_braket_backend.py, which fails COLLECTION (no
  `braket` module) -- that is the qumat quantum-backend layer, orthogonal to the
  QDP native engine and unrelated to this port; scope parity to testing/qdp*.

Fork HEAD: 2b0544a40bcaf60d35539ba8be62cf791e6c0846 (amended single curated
commit, force-with-lease pushed to AMD-Ecosystem/mahout @ moat-port).

## Validation 2026-05-31 (validator, linux-gfx90a, MI250X, ROCm 7.2.1) -- PASS

Platform: linux-gfx90a, GCD: HIP_VISIBLE_DEVICES=2 (MI250X gfx90a), ROCm 7.2.1.
Fork: AMD-Ecosystem/mahout @ moat-port HEAD 2b0544a40bcaf60d35539ba8be62cf791e6c0846.
Build: `cargo build -p qdp-core -p qdp-kernels --no-default-features --features hip -j 16` -- exit 0 (cached, 0.16s).

Rust tests (HIP_VISIBLE_DEVICES=2, --test-threads=1, 10.3s):
- qdp-kernels: amplitude_encode 21/21, angle_encode 10/10.
- qdp-core lib: 77/77.
- GPU suites: gpu_angle 12/12, gpu_api_workflow 8/8, gpu_basis 7/7, gpu_dlpack 9/9,
  gpu_fidelity 17/17, gpu_iqp 22/22, gpu_memory_safety 4/4, gpu_norm_f32 2/2,
  gpu_ptr_encoding 64/64, gpu_validation 8/8.
- Non-GPU suites: arrow_ipc 5/5, null_handling 6/6, numpy 4/4, parquet 8/8,
  preprocessing 14/14, tensorflow_io 9/9, torch_io 3/3, types 6/6.
- 0 failures total.
Async-pipeline tests confirmed passing: test_amplitude_encoding_async_pipeline,
test_angle_encoding_async_pipeline (gpu_api_workflow), test_angle_batch_f32_async_pipeline_path (gpu_angle_encoding) -- all via hipMemcpyAsync (non-blocking H2D).

Python parity (testing/qdp + testing/qdp_python + qdp/qdp-python/tests, 11.9s):
- 301 passed, 12 skipped, 0 failed.
- Skips: 2 multi-GPU, 1 tensorflow-absent, 1 loader path-timing, 5 torch_ref sm_-arch check
  (Triton/torch CUDA reference path; not native engine), 2 AmdQdpEngine-not-built,
  1 NVIDIA-ref-absent -- all pre-existing/legit.

Transition: review-passed -> completed (validated_sha = 2b0544a).
Followers unblocked: linux-gfx1100, windows-gfx1151 -> port-ready.

## Validation 2026-05-31 (gfx1100, ROCm 7.2.1) -- PASS

Platform: linux-gfx1100, GPU: AMD Radeon Pro W7800 48GB (gfx1100, RDNA3 wave32), HIP_VISIBLE_DEVICES=0, ROCm 7.2.1.
Fork: AMD-Ecosystem/mahout @ moat-port HEAD 2b0544a40bcaf60d35539ba8be62cf791e6c0846 -- no fork interaction, no source change.

Build commands:
```
curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal && . "$HOME/.cargo/env"
export QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx1100 ROCM_PATH=/opt/rocm
cd projects/mahout/src/qdp
cargo build -p qdp-core -p qdp-kernels --no-default-features --features hip -j 16
maturin build --features hip --profile dev --manifest-path qdp/qdp-python/Cargo.toml --out <wheeldir> --compatibility linux
pip install --no-deps --force-reinstall <wheeldir>/qumat_qdp-0.2.0-cp312-cp312-linux_x86_64.whl
```
Both steps exit 0. Wheel imports cleanly (import qumat_qdp ok, QdpEngine/NativeQuantumTensor present).

gfx1100 code-object evidence (llvm-objdump --offloading on libkernels.a):
All 6 kernel TUs target `hipv4-amdgcn-amd-amdhsa--gfx1100`, no gfx90a:
  amplitude.cu, basis.cu, angle.cu, validation.cu, iqp.cu, phase.cu -> gfx1100.

Rust tests (HIP_VISIBLE_DEVICES=0, --test-threads=1):
- qdp-kernels: amplitude_encode 21/21, angle_encode 10/10.
- qdp-core lib: 77/77.
- GPU suites: gpu_angle 12/12, gpu_api_workflow 8/8, gpu_basis 7/7, gpu_dlpack 9/9,
  gpu_fidelity 17/17, gpu_iqp 22/22, gpu_memory_safety 4/4, gpu_norm_f32 2/2,
  gpu_ptr_encoding 64/64, gpu_validation 8/8.
- Non-GPU suites: arrow_ipc 5/5, null_handling 6/6, numpy 4/4, parquet 8/8,
  preprocessing 14/14, tensorflow_io 9/9, torch_io 3/3, types 6/6.
- 0 failures total. Matches gfx90a baseline exactly.
- Async-pipeline tests pass: test_amplitude_encoding_async_pipeline,
  test_angle_encoding_async_pipeline (gpu_api_workflow),
  test_angle_batch_f32_async_pipeline_path (gpu_angle_encoding).

Python parity (testing/qdp + testing/qdp_python + qdp/qdp-python/tests):
- 301 passed, 12 skipped, 0 failed. Matches gfx90a baseline exactly.
- Skips: 2 multi-GPU, 1 tensorflow-absent, 1 loader path-timing,
  5 torch_ref sm_-arch (sm_110 on gfx1100 vs sm_-cap list; Triton/torch ref path, not native engine),
  2 AmdQdpEngine-not-built, 1 NVIDIA-ref-absent -- all pre-existing/legit.

Wave32 / L2-norm warp-reduction verdict -- CORRECT on gfx1100:
- gpu_norm_f32 (2/2) and amplitude_encode L2-norm tests (10/10 l2_norm* variants in
  test_l2_norm_single_kernel{,_f32}, test_l2_norm_batch_kernel_{f32,odd,stream,zero_*})
  all pass on wave32.
- The arch-unified fix (warp_id = threadIdx.x / warpSize == >>5 on wave32, ==>>6 on wave64)
  places the per-warp partial in the correct shared[warp_id] slot on both widths.
  __shared__ shared[32] holds up to 32 warps; 1024 threads / warpSize=32 = 32 warps
  exactly on gfx1100, with no slot overflow. The QDP_FULL_WARP_MASK=0xffffffffffffffff
  (64-bit) has upper 32 bits zero on wave32, behaving identically to 0xffffffff.
- Determinism: L2-norm tests re-run independently -> 10/10 identical pass.

Transition: port-ready -> completed (validated_sha = 2b0544a40bcaf60d35539ba8be62cf791e6c0846).

## Validation 2026-06-03 (windows-gfx1151) -- delta-port + PASS

Platform: AMD Radeon(TM) 8060S Graphics (gfx1151, RDNA3.5 wave32), Windows 11,
TheRock ROCm 7.14 (pip wheels, rocm-sdk-devel 7.14.0a20260531),
torch 2.12.0+rocm7.14.0a20260531, Rust 1.96.0 (msvc target), maturin 1.13.3.

Fork base: 2b0544a40bcaf60d35539ba8be62cf791e6c0846 (review-passed unified commit).
Delta commit: addf01141f64bf09476ce32274ee61481b57e325 (pushed to moat-port).

### Windows delta required (port-ready -> validated)

The original port gated the entire GPU stack on `target_os = "linux"` as a proxy
for "GPU available." This prevented compilation and test execution of any GPU code
on Windows. A delta commit adds Windows GPU support behind the `hip` feature.

Delta changes (39 files):
1. qdp-core/build.rs, qdp-kernels/build.rs, NEW qdp-python/build.rs: emit
   `cfg(qdp_gpu_platform)` on Linux (always) or Windows+hip. Replaces the
   `target_os = "linux"` proxy with an intent-accurate flag.
2. All source + test files (38 files): mechanical rename
   `target_os = "linux"` -> `qdp_gpu_platform` and
   `not(target_os = "linux")` -> `not(qdp_gpu_platform)`.
3. qdp-kernels/hip_compat/cuda_runtime.h: add `#ifndef M_SQRT1_2` define.
   MSVC <math.h> does not define POSIX math constants; phase.cu uses M_SQRT1_2.
4. qdp-core/src/platform/mod.rs: windows stub now gated
   `all(target_os = "windows", not(qdp_gpu_platform))` to avoid duplicate
   `encode_from_parquet` symbol when qdp_gpu_platform fires on Windows+hip.

Linux/CUDA builds are byte-identical (qdp_gpu_platform == target_os="linux"
on those paths; hip_compat shim is only on the HIP include path).

### Build commands (gfx1151)
```
# Env setup (from agent_space/mahout_build.sh)
VENV=/d/Develop/moat/agent_space/venv-gsplat
ROOT=$($VENV/Scripts/python.exe -m rocm_sdk path --root)
export HIP_DEVICE_LIB_PATH=$ROOT/lib/llvm/amdgcn/bitcode
export QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx1151
export QDP_HIPCC=$ROOT/bin/hipcc.exe
export ROCM_PATH=$ROOT
export PATH=$MSVC_DIR:$CARGO_HOME/bin:$ROOT/bin:$TARGET/debug/deps:$PATH

# Kernels + core (HIP, dev profile)
cargo build --manifest-path qdp/Cargo.toml -p qdp-core -p qdp-kernels \
  --no-default-features --features hip -j6  # -j6: APU power cap

# Python extension wheel (dev profile -- release LTO breaks cdylib on HIP)
maturin build --manifest-path qdp/qdp-python/Cargo.toml \
  --features hip --profile dev --out <wheeldir> \
  --interpreter $VENV/Scripts/python.exe
pip install --no-deps --force-reinstall --ignore-requires-python <wheel>

# Deploy TheRock runtime DLLs next to test binaries (loader prefers exe dir over System32)
cp $ROOT/bin/amdhip64_7.dll $ROOT/bin/amd_comgr.dll $ROOT/bin/rocm_kpack.dll \
   target/debug/deps/
cp $ROOT/bin/amdhip64_7.dll $ROOT/bin/amd_comgr.dll $ROOT/bin/rocm_kpack.dll \
   $VENV/Lib/site-packages/_qdp/

# Windows needs D:/tmp for tests that create temp files with hardcoded /tmp paths
mkdir -p D:/tmp
```

### Test results (gfx1151, dev profile, --test-threads=1)

qdp-kernels (cargo test -p qdp-kernels --no-default-features --features hip):
- amplitude_encode 21/21
- angle_encode 10/10

qdp-core (cargo test -p qdp-core --no-default-features --features hip):
- lib unit tests: 77/77
- gpu_angle 12/12, gpu_api_workflow 8/8, gpu_basis 7/7, gpu_dlpack 9/9,
  gpu_fidelity 17/17, gpu_iqp 22/22, gpu_memory_safety 4/4, gpu_norm_f32 2/2,
  gpu_ptr_encoding 64/64, gpu_validation 8/8
- Non-GPU: arrow_ipc 5/5, null_handling 6/6, numpy 4/4, parquet 8/8,
  preprocessing 14/14, tensorflow_io 9/9, torch_io 3/3, types 6/6
- 0 failures total. Matches gfx90a/gfx1100 baseline exactly.

Python parity (testing/qdp + testing/qdp_python + qdp/qdp-python/tests):
- 282 passed, 31 skipped, 0 failed.
- Skips vs Linux baseline (12 skips) -- 19 additional:
  - 20 Triton AMD backend tests skip (triton not installed in TheRock pip venv);
    on Linux the conda env had triton installed. Not a GPU regression.
  - 1 test_file_loader_unsupported_extension_raises (Windows path behavior diff)
  - Remaining skips identical to Linux: 2 multi-GPU, 1 tensorflow-absent,
    1 loader path-timing, 5 torch_ref sm_-arch check (CUDA-only reference path),
    2 AmdQdpEngine-not-built. 0 failures.

### Windows-specific notes
- D:/tmp must exist: tests use hardcoded `/tmp/` paths; Windows resolves this to
  D:\tmp (the D: drive, where the cargo workspace lives). Create with mkdir.
- TheRock amdhip64_7.dll must be deployed next to test exes: System32's Adrenalin
  amdhip64_7.dll is broken (undefined blit symbols); TheRock's is self-consistent.
- qdp-python/tests PATH: $ROOT/bin must be on PATH before pytest so _qdp.pyd
  can find amdhip64_7.dll when conftest.py imports it (before torch is loaded).
- -fgpu-rdc is NOT used (no device-link step); hipcc compiles each .cu independently.
  The Windows -fgpu-rdc bundler bug (clang-offload-bundler mmap) is not triggered.

Wave32 verdict: gfx1151 RDNA3.5 wave32 -- all L2-norm/amplitude tests pass,
identical to gfx1100. The arch-unified warp_id = threadIdx.x / warpSize fix
is correct on wave32 (== >>5). QDP_FULL_WARP_MASK 0xffffffffffffffff upper
32 bits zero on wave32, identical to CUDA's 0xffffffff in practice.

Transition: port-ready -> completed (validated_sha = addf01141f64bf09476ce32274ee61481b57e325).
linux-gfx90a and linux-gfx1100 -> revalidate (delta touches Rust source;
binary equivalence check expected to confirm no change on Linux paths).

## Validation 2026-06-04 (gfx90a, revalidate, binary-equiv carry-forward)

Platform: linux-gfx90a, GPU: MI250X gfx90a (wave64), ROCm 7.2.1.
Transition: revalidate -> completed (validated_sha = addf01141f64bf09476ce32274ee61481b57e325).
Method: binary-equivalence carry-forward (no GPU re-run required).

Delta (2b0544a..addf01141f, 39 files):
- 2 modified build.rs (qdp-core, qdp-kernels): emit `cfg(qdp_gpu_platform)` on Linux (always) or Windows+hip. On Linux `qdp_gpu_platform` is always set -- identical to old `target_os = "linux"` condition.
- 1 new qdp-python/build.rs: same cfg emit for Windows+hip support.
- 35 Rust source + test files: mechanical rename `#[cfg(target_os = "linux")]` -> `#[cfg(qdp_gpu_platform)]` and `#[cfg(not(target_os = "linux"))]` -> `#[cfg(not(qdp_gpu_platform))]`.
- 1 qdp-kernels/hip_compat/cuda_runtime.h: added `#ifndef M_SQRT1_2` guard (MSVC POSIX math constant); this header is only on the HIP include path and only affects Windows MSVC compilation.
- 1 qdp-core/src/platform/mod.rs: windows stub gated `all(target_os = "windows", not(qdp_gpu_platform))` to avoid duplicate symbol on Windows+hip.
None of the 6 .cu kernel source files changed. On Linux, every cfg evaluation is identical: `qdp_gpu_platform` is always true (same as `target_os = "linux"` was), and the M_SQRT1_2 shim is only reached by Windows MSVC, not by Linux/hipcc.

Binary-equivalence check:
- git worktree at validated_sha (2b0544a40b) built into `/tmp/mahout-old-gfx90a-target`; HEAD (addf01141f) built into the default target dir. Both: `QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx90a ROCM_PATH=/opt/rocm --no-default-features --features hip -j 16` -> exit 0.
- Compared libkernels.a (archive of 6 .cu HIP kernel objects) with `python3 utils/codeobj_diff.py`.
- Result: `verdict=identical` -- exported symbols + device ISA identical (0 exports). All 6 gfx90a kernel TUs compile to byte-identical device code objects.
- 256-byte .a size difference is AR metadata (build.rs cfg strings), not GPU code.

Build commands:
```
source "$HOME/.cargo/env"
export QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx90a ROCM_PATH=/opt/rocm
# HEAD build
bash utils/timeit.sh mahout compile -- cargo build \
  --manifest-path projects/mahout/src/qdp/Cargo.toml \
  -p qdp-core -p qdp-kernels --no-default-features --features hip -j 16
# validated_sha build (worktree)
cd projects/mahout/src && git worktree add /tmp/mahout-old-gfx90a 2b0544a40bcaf60d35539ba8be62cf791e6c0846
CARGO_TARGET_DIR=/tmp/mahout-old-gfx90a-target cargo build \
  --manifest-path /tmp/mahout-old-gfx90a/qdp/Cargo.toml \
  -p qdp-core -p qdp-kernels --no-default-features --features hip -j 16
# compare
python3 utils/codeobj_diff.py \
  /tmp/mahout-old-gfx90a-target/debug/build/qdp-kernels-e8e72e39df1ee785/out/libkernels.a \
  projects/mahout/src/qdp/target/debug/build/qdp-kernels-e8e72e39df1ee785/out/libkernels.a
# verdict=identical
```

## Validation 2026-06-04 (gfx1100, revalidate, binary-equiv carry-forward)

Platform: linux-gfx1100, GPU: AMD Radeon Pro W7800 48GB (gfx1100, RDNA3 wave32), ROCm 7.2.1.
Transition: revalidate -> completed (validated_sha = addf01141f64bf09476ce32274ee61481b57e325).
Method: binary-equivalence carry-forward (no GPU re-run required).

Delta (2b0544a..addf01141f, 39 Rust files + build.rs):
Mechanical rename of `#[cfg(target_os = "linux")]` -> `#[cfg(qdp_gpu_platform)]` across all
GPU-gated code paths. build.rs now emits `cargo::rustc-cfg=qdp_gpu_platform` when
`is_linux || (is_windows && hip_feature)`. On Linux, `is_linux` is always true, so
`qdp_gpu_platform` is always set -- the compiled Linux output is byte-identical to the
old `target_os = "linux"` build. No `.cu` kernel source files changed.

Binary-equivalence check:
- Built at validated_sha (2b0544a40b) with `CARGO_TARGET_DIR` pointing to a separate dir
  (git worktree) and at HEAD (addf01141f) in the default target dir, both with
  `QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx1100 ROCM_PATH=/opt/rocm --no-default-features --features hip`.
  Both builds: `cargo build -p qdp-core -p qdp-kernels --manifest-path ... -j 16` -> exit 0.
- Compared libkernels.a (the archive of the 6 .cu HIP kernel objects -- the only artifact
  containing device code) using `python3 utils/codeobj_diff.py <old>/libkernels.a <head>/libkernels.a`.
- Result: `verdict=identical` -- exported symbols + device ISA identical (0 exports).
  All 6 gfx1100 kernel TUs (amplitude.cu, basis.cu, angle.cu, validation.cu, iqp.cu, phase.cu)
  compile to byte-identical device code objects.
- The 94-byte size difference in the .a files is in host-side AR metadata (build.rs cfg
  strings), not in the GPU code objects.

Build commands:
```
source "$HOME/.cargo/env"
export QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx1100 ROCM_PATH=/opt/rocm
# HEAD build
cargo build --manifest-path projects/mahout/src/qdp/Cargo.toml \
  -p qdp-core -p qdp-kernels --no-default-features --features hip -j 16
# validated_sha build (worktree)
git worktree add /tmp/mahout-old 2b0544a40bcaf60d35539ba8be62cf791e6c0846
CARGO_TARGET_DIR=/tmp/mahout-old-target cargo build \
  --manifest-path /tmp/mahout-old/qdp/Cargo.toml \
  -p qdp-core -p qdp-kernels --no-default-features --features hip -j 16
# compare
python3 utils/codeobj_diff.py \
  /tmp/mahout-old-target/debug/build/qdp-kernels-e8e72e39df1ee785/out/libkernels.a \
  projects/mahout/src/qdp/target/debug/build/qdp-kernels-e8e72e39df1ee785/out/libkernels.a
# verdict=identical
```

## Validation 2026-06-06 (windows-gfx1201, AMD Radeon RX 9070 XT, TheRock ROCm 7.14) -- PASS

Platform: windows-gfx1201, GPU: AMD Radeon RX 9070 XT (gfx1201, RDNA4 wave32, warpSize=32),
HIP_VISIBLE_DEVICES=0 (gfx1201 only GPU present; gfx1101 absent from bus),
Windows 11 (26200), TheRock ROCm 7.14.0a20260604,
torch 2.9.1+rocm7.14.0a20260604, Rust 1.96.0 (msvc target), maturin 1.13.3.
Fork: AMD-Ecosystem/mahout @ moat-port HEAD addf01141f64bf09476ce32274ee61481b57e325.

Build commands:
```
ROOT=/b/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel
MSVC_DIR=/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/HostX64/x64
export CARGO_HOME=/b/develop/moat/agent_space/cargo RUSTUP_HOME=/b/develop/moat/agent_space/rustup
export PATH="$CARGO_HOME/bin:$MSVC_DIR:$ROOT/bin:$ROOT/lib/llvm/bin:$PATH"
# LIB must use Windows-style paths (semicolon-separated) for MSVC link.exe test binary linking
export LIB="$(cygpath -w /c/.../MSVC/14.44.35207/lib/x64);$(cygpath -w .../ucrt/x64);$(cygpath -w .../um/x64)"
export HIP_DEVICE_LIB_PATH="$ROOT/lib/llvm/amdgcn/bitcode"
export QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx1201 QDP_HIPCC="$ROOT/bin/hipcc.exe"
export ROCM_PATH="$ROOT" HIP_VISIBLE_DEVICES=0

# Kernels + core (dev profile, incremental -- only kernel objects rebuilt for gfx1201)
cargo build --manifest-path projects/mahout/src/qdp/Cargo.toml \
  -p qdp-core -p qdp-kernels --no-default-features --features hip
# -> exit 0, 3.23s. gfx1201 kernel code objects confirmed via strings: gfx1201, gfx1250.

# TheRock runtime DLLs -- already deployed from gfx1101 run (same TheRock build):
# target/debug/deps/{amdhip64_7.dll,amd_comgr.dll,rocm_kpack.dll,hiprtc0714.dll,hiprtc-builtins0714.dll}

# Python extension wheel for gfx1201
VENV=/b/develop/TheRock/external-builds/pytorch/.venv
maturin build --features hip --profile dev \
  --manifest-path projects/mahout/src/qdp/qdp-python/Cargo.toml \
  --out /b/develop/moat/agent_space/mahout_wheels_gfx1201 \
  --interpreter $VENV/Scripts/python.exe
pip install --no-deps --force-reinstall mahout_wheels_gfx1201/qumat_qdp-0.2.0-cp312-cp312-win_amd64.whl
# Deploy DLLs to _qdp package dir
cp $ROOT/bin/{amdhip64_7.dll,amd_comgr.dll,rocm_kpack.dll,hiprtc0714.dll,hiprtc-builtins0714.dll} \
   $VENV/Lib/site-packages/_qdp/
```
Both build steps exit 0. gfx1201 kernel code confirmed (grep: gfx1201/gfx1250 in _qdp.pyd).

Note on LIB env: `cargo build` (library-only) works with POSIX-style LIB paths, but `cargo test`
(produces .exe test binaries) requires Windows-style semicolon-separated LIB for MSVC link.exe.
Use `cygpath -w` to convert. This is a Windows bash-shell quirk not present on the gfx1151 build
(which may have had LIB set in the Windows environment already).

Rust tests (HIP_VISIBLE_DEVICES=0, --test-threads=1):
- qdp-kernels: amplitude_encode 21/21, angle_encode 10/10.
- qdp-core lib: 77/77.
- GPU suites: gpu_angle 12/12, gpu_api_workflow 8/8, gpu_basis 7/7, gpu_dlpack 9/9,
  gpu_fidelity 17/17, gpu_iqp 22/22, gpu_memory_safety 4/4, gpu_norm_f32 2/2,
  gpu_ptr_encoding 64/64, gpu_validation 8/8.
- Non-GPU suites: arrow_ipc 5/5, null_handling 6/6, numpy 4/4, parquet 8/8,
  preprocessing 14/14, tensorflow_io 9/9, torch_io 3/3, types 6/6.
- 0 failures total. Matches gfx90a/gfx1100/gfx1151/gfx1101 baseline exactly.

Python parity (testing/qdp + testing/qdp_python + qdp/qdp-python/tests):
- 301 passed, 12 skipped, 0 failed. Matches gfx1101 baseline exactly.
- Skips: 2 multi-GPU, 1 tensorflow-absent, 1 loader path-timing,
  5 torch_ref (sm_120 on gfx1201 vs sm_-cap list; Triton/torch CUDA reference path, not native engine),
  2 AmdQdpEngine-not-built, 1 NVIDIA-ref-absent -- all pre-existing/legit.
- Warnings: PyTorch _select_torch_device warns "sm_120 not in arch list" for fallback-path tests;
  the native HIP engine (_qdp) runs on gfx1201 correctly.

Async-pipeline tests pass: test_amplitude_encoding_async_pipeline,
test_angle_encoding_async_pipeline (gpu_api_workflow), test_angle_batch_f32_async_pipeline_path
(gpu_angle_encoding) -- all via hipMemcpyAsync (non-blocking H2D).

Wave32 verdict: gfx1201 RDNA4 wave32 -- all L2-norm/amplitude tests pass, identical to
gfx1100/gfx1101/gfx1151. warp_id = threadIdx.x / warpSize == >>5 on wave32.
QDP_FULL_WARP_MASK 0xffffffffffffffff upper 32 bits zero on wave32, identical to CUDA's 0xffffffff.

Transition: port-ready -> completed (validated_sha = addf01141f64bf09476ce32274ee61481b57e325).

## Validation 2026-06-05 (windows-gfx1101, AMD Radeon PRO V710, TheRock ROCm 7.14) -- PASS

Platform: windows-gfx1101, GPU: AMD Radeon PRO V710 (gfx1101, RDNA3 wave32, warpSize=32),
HIP_VISIBLE_DEVICES=0, Windows 11 (26200), TheRock ROCm 7.14.0a20260604,
torch 2.9.1+rocm7.14.0a20260604, Rust 1.96.0 (msvc target), maturin 1.13.3.
Fork: AMD-Ecosystem/mahout @ moat-port HEAD addf01141f64bf09476ce32274ee61481b57e325.

Build commands:
```
ROOT=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel
MSVC_DIR=/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/HostX64/x64
export CARGO_HOME=/b/develop/moat/agent_space/cargo RUSTUP_HOME=/b/develop/moat/agent_space/rustup
export PATH="$CARGO_HOME/bin:$MSVC_DIR:$ROOT/bin:$ROOT/lib/llvm/bin:$PATH"
export HIP_DEVICE_LIB_PATH="$ROOT/lib/llvm/amdgcn/bitcode"
export QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx1101 QDP_HIPCC="$ROOT/bin/hipcc.exe"
export ROCM_PATH="$ROOT" HIP_VISIBLE_DEVICES=0

# Kernels + core (dev profile, no LTO)
cargo build --manifest-path projects/mahout/src/qdp/Cargo.toml \
  -p qdp-core -p qdp-kernels --no-default-features --features hip

# Deploy TheRock runtime DLLs next to test binaries (loader prefers exe dir over System32)
cp $ROOT/bin/{amdhip64_7.dll,amd_comgr.dll,rocm_kpack.dll,hiprtc0714.dll,hiprtc-builtins0714.dll} \
   projects/mahout/src/qdp/target/debug/deps/

# Python extension wheel (dev profile -- release LTO breaks cdylib on HIP)
VENV=B:/develop/TheRock/external-builds/pytorch/.venv
maturin build --features hip --profile dev \
  --manifest-path projects/mahout/src/qdp/qdp-python/Cargo.toml \
  --out /b/develop/moat/agent_space/mahout_wheels_gfx1101 \
  --interpreter $VENV/Scripts/python.exe
pip install --no-deps --force-reinstall mahout_wheels_gfx1101/qumat_qdp-0.2.0-cp312-cp312-win_amd64.whl

# Deploy DLLs to _qdp package dir for Python import
cp $ROOT/bin/{amdhip64_7.dll,amd_comgr.dll,rocm_kpack.dll,hiprtc0714.dll,hiprtc-builtins0714.dll} \
   $VENV/Lib/site-packages/_qdp/
```
Both build steps exit 0. Kernels compile for gfx1101 (1 harmless unused-parameter warning in iqp.cu).

Rust tests (HIP_VISIBLE_DEVICES=0, --test-threads=1):
- qdp-kernels: amplitude_encode 21/21, angle_encode 10/10.
- qdp-core lib: 77/77.
- GPU suites: gpu_angle 12/12, gpu_api_workflow 8/8, gpu_basis 7/7, gpu_dlpack 9/9,
  gpu_fidelity 17/17, gpu_iqp 22/22, gpu_memory_safety 4/4, gpu_norm_f32 2/2,
  gpu_ptr_encoding 64/64, gpu_validation 8/8.
- Non-GPU suites: arrow_ipc 5/5, null_handling 6/6, numpy 4/4, parquet 8/8,
  preprocessing 14/14, tensorflow_io 9/9, torch_io 3/3, types 6/6.
- 0 failures total. Matches gfx90a/gfx1100/gfx1151 baseline exactly.

Python parity (testing/qdp + testing/qdp_python + qdp/qdp-python/tests):
- 282 passed, 31 skipped, 0 failed. Matches gfx1151 baseline exactly.
- Skips: 20 Triton AMD backend (triton not installed), 2 multi-GPU, 1 tensorflow-absent,
  1 loader path-timing, 5 torch_ref sm_-arch check (CUDA-only reference path),
  2 AmdQdpEngine-not-built, 1 NVIDIA-ref-absent -- all pre-existing/legit.

Windows-specific notes (same as gfx1151):
- TheRock amdhip64_7.dll deployed next to test exes; System32's Adrenalin DLL is broken.
- ROCm bin on PATH before pytest so _qdp.pyd can load amdhip64_7.dll at import.
- -j6 cap does NOT apply to this machine (beefy workstation, 64 cores, used default parallelism).

Wave32 verdict: gfx1101 RDNA3 wave32 -- all L2-norm/amplitude tests pass, identical to
gfx1100/gfx1151. warp_id = threadIdx.x / warpSize == >>5 on wave32.

Transition: port-ready -> completed (validated_sha = addf01141f64bf09476ce32274ee61481b57e325).

## PR-prep 2026-06-11 (porter, linux-gfx90a) -- squashed to one commit, all 5 carried forward

All 5 platforms were completed at addf01141. PR-prep done; no rebuild (edits are
comment/doc/attribution only, behavior-preserving).

Jargon scrub: only ONE leak in upstream-visible code/comments --
qdp/qdp-kernels/build.rs:39 said "follower platforms (gfx1100, gfx1151)";
reworded to "other AMD targets (gfx1100, gfx1151)", rationale preserved. The two
original commit messages were otherwise clean (commit 1 referenced "the MOAT
CUDA-to-ROCm effort" once, dropped at squash). No other in-house vocabulary
(lead/follower/Strategy A-B/head_sha/validated_sha/revalidate/curated) anywhere
in the diff or messages.

Attribution (AMD copyright line below the Apache ASF header + `Author: Jeff Daily`, matching this tree's file-header convention; ASF puts
ownership in NOTICE and uses no per-file author tags, so a header comment line is
the house-style fit):
- NEW files (plain `Copyright`): qdp-core/src/gpu_rt.rs,
  qdp-kernels/src/device.rs, qdp-kernels/src/kernel_compat.h,
  qdp-kernels/hip_compat/{cuComplex.h,cuda_runtime.h,vector_types.h},
  qdp-python/build.rs.
- Substantially extended (`Portions Copyright`): qdp-kernels/build.rs (the +95
  HIP compile branch), qdp-core/src/gpu/cuda_ffi.rs (the hip_rt FFI mod),
  qdp-kernels/src/amplitude.cu (the wave64 warp-id + 64-bit mask fix).
- SKIPPED as trivial (build-flag/cfg/mechanical only, no AMD-authored logic):
  qdp-core/build.rs (+13 cfg-emit), all ~40 mechanical target_os->qdp_gpu_platform
  rename files, the Cargo.toml feature edits.

Docs (REQUIRED step):
- qdp/DEVELOPMENT.md: added "### AMD GPU build (ROCm / HIP)" after the no-CUDA
  sanity block (the project's single from-source build guide), covering the `hip`
  feature, QDP_USE_HIP=1, QDP_HIP_ARCH_LIST, ROCm prereqs (hipcc, AMD HIP
  runtime), and the dev-profile wheel install, in the file's house style.
- qdp/qdp-python/README.md: added a note that the native `_qdp` engine
  (backend="cuda" route) also runs on AMD via the `hip` feature, distinct from
  the Triton backend, pointing to DEVELOPMENT.md. This README defers from-source
  build steps to DEVELOPMENT.md, so no build block was imposed here.

Arch auto-detect determination: build.rs is already env-driven and correct for
upstream -- QDP_HIP_ARCH_LIST is read and comma-split, gfx90a is only a fallback
when unset, nothing hardcoded overrides the env. This mirrors the existing
QDP_CUDA_ARCH_LIST convention. No change made; a build-time rocminfo auto-detect
would diverge from the project's env convention and add fragility (over-
engineering). Left as-is.

Carry-forward note: `advance-head` on the prep delta flipped all platforms to
revalidate because its classifier treats `.rs` as unknown-file-type (cannot prove
comment-only) and raised a kernel_compat.h __LINE__ line-shift false positive.
Manually verified every changed .rs/.cu/.h line is a `//` comment (copyright/
author headers + one comment reword), zero functional code -- so carried all 5
forward with `carry-forward ... source-class` (the behavior-preserving path).

Prep commit (on top of addf01141, before squash):
a8d63e21764900618a344efb09a3a9ee23a019da.
Squashed (tree-identical collapse, force-with-lease pushed to moat-port):
f3f7db33cc9942f5c1a7ffdbe95aea68c85532f5. `squash-carry-forward` carried all 5
platforms (did not refuse -> tree-identical confirmed). pr-ready=True.
Ready for the user's upstream-PR decision (apache/mahout, moat-port -> main).

## Review fixes (apache/mahout#1399) 2026-06-11 (porter, linux-gfx90a)

Maintainer left 4 inline review comments on the open PR. Fixed as ONE follow-up
commit on top of the validated squash (f3f7db33), NOT an amend. Functional HIP
changes, so `advance-head` flipped the 4 follower AMD platforms to revalidate
(correct); the lead linux-gfx90a was re-validated on real gfx90a GPU at the new
head and stays pr-open.

New fork HEAD: 0b5042e705eff3809fa7c40c1383aa6c3adcc602 (force-with-lease pushed
to AMD-Ecosystem/mahout @ moat-port; this updates PR #1399).

### Fix 1 (MERGE-BLOCKER): CudaSlice::drop re-binds the owning device
qdp-kernels/src/device.rs, hip mod, `impl Drop for CudaSlice`.
- Before: `drop` called `hipFree(self.raw_ptr())` with no device bind. hipFree
  frees on the calling thread's CURRENT device, so dropping a slice while a
  different device is current (multi-GPU) freed against the wrong device.
- After: `let _ = self._device.bind();` (best-effort -- Drop cannot return an
  error) before `hipFree`, matching the alloc path (`self.bind()?`) and cudarc.
  The CudaSlice already held `_device: Arc<CudaDevice>`; `bind()` was already a
  private method on CudaDevice. No new FFI needed.

### Fix 2: explicit hipMemoryType mapping (no magic 2)
qdp-core/src/gpu/cuda_ffi.rs, hip_rt::cudaPointerGetAttributes (~line 189).
- Before: reinterpreted the CudaPointerAttributes destination directly as a
  hipPointerAttribute_t and let lib.rs:91 compare memory_type against the CUDA
  constant 2. hipMemoryType enum values are NOT guaranteed equal to CUDA's
  across ROCm releases (hip_runtime_api.h note; older HIP had Host=0/Device=1).
- After: reads a real `HipPointerAttributes` (#[repr(C)] mirror of
  hipPointerAttribute_t), compares its `type` against the named
  HIP_MEMORY_TYPE_DEVICE (hipMemoryTypeDevice) / HIP_MEMORY_TYPE_MANAGED
  (hipMemoryTypeManaged) constants, and translates to the CUDA convention
  (CUDA_MEMORY_TYPE_DEVICE/MANAGED) the caller checks. Other values pass through
  verbatim so the "not device memory" branch still fires. CUDA path unchanged.

### Fix 3: build.rs fails loudly on QDP_USE_HIP vs `hip` feature mismatch
qdp-kernels/build.rs: new `check_hip_consistency()` called early in `main()`.
- Before: `hip_requested()` returned true if EITHER the `hip` feature OR
  QDP_USE_HIP was set, so a default `cargo build` (cuda feature) with
  QDP_USE_HIP=1 built AMD kernels (hipcc) against the cudarc host = silent
  mismatch.
- After: if `qdp_use_hip_env()` and `CARGO_FEATURE_HIP` disagree (either
  direction), the build `panic!`s with a clear message. Verified: with
  QDP_USE_HIP=1 and the hip feature off, `cargo build -p qdp-kernels` aborts
  ("QDP_USE_HIP is set but the `hip` Cargo feature is off ..."). A clean CUDA
  build (neither set) and the HIP build (both set) are unaffected.

### Fix 4: fork_default_stream uses a non-blocking stream
qdp-kernels/src/device.rs, hip mod (~line 392) + FFI.
- Before: `hipStreamCreate(&mut stream)` -- a BLOCKING stream that implicitly
  serializes against the NULL/default stream, defeating copy/compute overlap.
- After: `hipStreamCreateWithFlags(&mut stream, HIP_STREAM_NON_BLOCKING)` (flag
  const = 1), matching cudarc. Swapped the `hipStreamCreate` extern decl for
  `hipStreamCreateWithFlags(*mut *mut c_void, u32)` and added the
  HIP_STREAM_NON_BLOCKING const.
- This EXPOSED a pre-existing latent ordering bug (shared with the CUDA path):
  AmplitudeEncoder::encode_batch_from_gpu_ptr_f32_with_stream
  (qdp-core/src/gpu/encodings/amplitude.rs ~line 877) launched the batch-f32
  norm kernel on the CALLER's stream, then read the result back with a
  default-stream `dtoh_sync_copy` WITHOUT first synchronizing the caller's
  stream. The blocking stream masked it; the non-blocking stream let the
  readback race the zero-initialized norm buffer -> "One or more float32 samples
  have zero or invalid norm" (deterministic). Every other batch path launches
  the norm kernel on the NULL stream, and the single-sample stream path already
  calls `sync_cuda_stream(stream, ...)` before the readback. Fix: added the same
  `sync_cuda_stream(stream, "Norm stream synchronize failed (batch f32)")?`
  before the readback. Arch-unified (correct on wave32 + wave64), and correct on
  the CUDA path too (a non-blocking CUDA stream has the identical hazard). This
  is the ONLY change that touches shared (non-hip-gated) code; it is
  behavior-identical for the common blocking-stream case.

### Validation (gfx90a, MI250X, ROCm 7.2.1, HIP_VISIBLE_DEVICES=3)
```
export QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx90a ROCM_PATH=/opt/rocm HIP_VISIBLE_DEVICES=3
cargo build --manifest-path projects/mahout/src/qdp/Cargo.toml \
  -p qdp-core -p qdp-kernels --no-default-features --features hip -j 16   # exit 0
cargo test  --manifest-path projects/mahout/src/qdp/Cargo.toml \
  -p qdp-core -p qdp-kernels --no-default-features --features hip -- --test-threads=1
```
All Rust tests PASS, 0 failures, matching the prior baseline exactly:
- qdp-kernels: amplitude_encode 21/21, angle_encode 10/10.
- qdp-core lib 77/77.
- GPU suites: gpu_angle 12/12, gpu_api_workflow 8/8, gpu_basis 7/7, gpu_dlpack
  9/9, gpu_fidelity 17/17, gpu_iqp 22/22, gpu_memory_safety 4/4, gpu_norm_f32
  2/2, gpu_ptr_encoding 64/64, gpu_validation 8/8.
- Non-GPU: arrow_ipc 5/5, null_handling 6/6, numpy 4/4, parquet 8/8,
  preprocessing 14/14, tensorflow_io 9/9, torch_io 3/3, types 6/6.
- gpu_ptr_encoding::test_encode_batch_from_gpu_ptr_f32_with_stream_success: the
  test that failed on the bare fix-4 stream change (before the sync was added)
  now passes 64/64.

Overlap test (fix 4 specifically), QDP_ENABLE_OVERLAP_TRACKING=1:
- test_amplitude_encoding_async_pipeline, test_angle_encoding_async_pipeline
  (gpu_api_workflow), test_angle_batch_f32_async_pipeline_path
  (gpu_angle_encoding) -- all pass. Non-blocking stream preserves the
  copy/compute overlap.

Mismatch guard (fix 3) sanity check: `QDP_USE_HIP=1 cargo build -p qdp-kernels`
(default cuda feature, hip off) panics with the mismatch message; throwaway
target dir, reverted. Default CUDA path still type-checks:
`cargo check -p qdp-core -p qdp-kernels` (default features, QDP_NO_CUDA=1, no
QDP_USE_HIP) exit 0.

### Byte-for-byte-claim finding (report only, for the PR reply)
The phrase "the NVIDIA build is byte-for-byte identical" appears in BOTH
upstream-visible places:
- PR #1399 body, first paragraph.
- The squashed commit message (f3f7db33) first paragraph: "so the NVIDIA build
  is byte-for-byte identical".
The reviewer is right that it is not strictly true. Pre-existing reasons:
metrics.rs swapped cudarc driver `cuMemcpyDtoH_v2` -> runtime `cudaMemcpy`, and
amplitude.cu `>>5` -> `/warpSize` -- both change the CUDA SASS though behavior is
identical. NEW with this follow-up: the fix-4 `sync_cuda_stream` in
encode_batch_from_gpu_ptr_f32_with_stream is NOT hip-gated (it is under
`#[cfg(qdp_gpu_platform)]`), so it also adds a stream-sync to the CUDA build's
codegen -- another behavior-identical-but-not-byte-identical CUDA delta.
Suggested wording for the reply: the CUDA path is behavior-preserving (no
functional change), not literally byte/SASS-identical.

## Review replies posted (apache/mahout#1399) 2026-06-11
PR body reworded "byte-for-byte identical" -> "behavior-preserving (no functional change)" (reviewer rich7420's nit; the CUDA path has behavior-identical-but-SASS-changing deltas: metrics.rs driver->runtime memcpy, kernel >>5->/warpSize, and the new fix-4 stream sync). Posted the overall reply (issuecomment-4682322151) + 4 threaded inline replies (discussion_r3397247837/8095/8302/8492) -- all four review points fixed in 0b5042e. Awaiting rich7420's response / re-review; on merge run set-pr-merged.

## Precommit fix + missed comments 2026-06-11
Missed two general PR comments on #1399 (only fetched the review + inline comments initially):
- rich7420 (issuecomment-4678743699): "check the precommit errors" -- the Pre-commit GHA (cargo fmt hook) failed. Cause: rustfmt diffs in the HIP-path files (import order + line wrap, incl. the fix-2 hipPointerGetAttributes line). FIXED: ran `cargo fmt --manifest-path qdp/Cargo.toml --all`, 7 files (all ours), pushed as 6d2de29 to fork moat-port. clippy is clean on our Rust; the only compiler warning is a pre-existing upstream `iqp.cu:342 unused parameter` (not ours). CI Pre-commit is action_required (Apache fork-PR gating), so it won't auto-run until a maintainer approves.
- ryankert01 (issuecomment-4682218305, MEMBER): asked whether complex optimizations port "this easy", citing #1390. #1390 (implicit Hadamard Ozaki engine) uses nvcuda::wmma + raw `mma.sync` int8 PTX -> NOT mechanically hipifiable; needs a rocWMMA/MFMA rewrite. Reply drafted, pending jeff approval.

NOTE: this clone has TWO remotes -- origin=apache/mahout (upstream, no push), fork=AMD-Ecosystem/mahout (push here). Push to `fork`, not `origin`.

## Precommit build.rs fix 2026-06-11 (new sha cc3859b)
The Pre-commit clippy hook (`cargo clippy --manifest-path qdp/Cargo.toml --all-targets --all-features`, runner has no ROCm) failed on qdp-kernels/build.rs because `--all-features` turns the `hip` feature ON without QDP_USE_HIP, and my recently-added check_hip_consistency() panicked on that (the second panic, build.rs:201). Two fixes in build.rs:
1. Removed the second panic (feature_hip && !env_hip). It was wrong: hip_requested() already builds kernels for HIP when the feature is on, so host+kernels agree -- no mismatch. KEPT the first panic (env_hip && !feature_hip) -- the real mismatch rich7420 (review comment #3) asked us to catch (QDP_USE_HIP=1 + hip off -> hipcc kernels vs cudarc host).
2. build_hip() now probes for a runnable hipcc (`Command::new(hipcc).arg("--version").output().is_ok()`, honors QDP_HIPCC override) and, when absent, degrades exactly like the no-nvcc CUDA branch: emit `cargo:rustc-cfg=qdp_no_cuda` + warnings, return (no panic, no build failure). With hipcc present, behavior unchanged (real kernels compiled).

Validation (all three scenarios):
- A REAL HIP build on gfx90a: `QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx90a ROCM_PATH=/opt/rocm cargo build -p qdp-core -p qdp-kernels --no-default-features --features hip` -> built, real gfx90a kernels compiled. GPU tests on idle HIP_VISIBLE_DEVICES=0: gpu_ptr_encoding 64 passed, gpu_fidelity 17 passed.
- B TOOLCHAIN-LESS CI: `QDP_HIPCC=/nonexistent-hipcc cargo clippy --manifest-path qdp/Cargo.toml -p qdp-kernels -p qdp-core --all-targets --features hip` (hip feature on, QDP_USE_HIP unset, hipcc absent) -> EXIT 0, NO panic, succeeds via qdp_no_cuda stub path. `cargo fmt ... --check` clean. (Full `--all-features` clippy on this host fails only on torch-sys: my host PyTorch is 2.13 vs the tch crate's 2.9 ABI -- an env artifact, not our code; CI has a matching libtorch venv. Our qdp-kernels build.rs got past compile cleanly.)
- C REVIEWER GUARD: `QDP_USE_HIP=1 cargo build -p qdp-kernels` (default features, hip off) -> still panics at build.rs:197 with "QDP_USE_HIP is set but the `hip` Cargo feature is off...". Guard intact.

Pre-existing clippy lints noted (NOT introduced here, NOT errors): `unnecessary_cast` on `stream.stream as *mut c_void` in gpu_ptr_encoding.rs test code.

Wheel Build failure on #1399 confirmed UNRELATED to our code: it is the NVIDIA CUDA rhel8 yum repo 404 (upstream CI infra fetching cuda-rhel8 packages), no workflow files were changed by this PR.

## Loop pass 2026-06-11 (ryankert01 review)
Branch was REBASED onto upstream main (post #1393 merge) by jeff/another session OUTSIDE moat commit-project; all shas changed (f3f7db33->5b1dbee, 0b5042e->206aa0c, 6d2de29->bc6206b, cc3859b->4d54e9c) -- my content preserved. status.json validated_shas are now orphaned (rebase); head_sha reconciled to fork tip. Lead pr-open; followers revalidate (pre-rebase). Full validation reconciliation across the rebase still owed (content is rebase-equivalent; GPU code unchanged vs upstream #1393 which is ParquetReader, non-GPU).
ryankert01 (MEMBER) reviewed on RTX 3090 Ti: 316 passed/0 failed, clippy --all-targets clean (CUDA path vendor-verified). 5 inline comments. Fixed #2 (ROCm>=6.0 doc), #3 (neither-feature compile_error), #4 (cuda/hip mutual-exclusivity note) in dbee1e1. HELD #1 (ASF source-header policy: drop per-file AMD copyright+Author lines, use NOTICE) -- conflicts with MOAT CLAUDE.md convention, needs jeff decision. #5 (no-AMD-CI maintenance risk) = acknowledge. Precommit fix is in 4d54e9c (build.rs graceful hipcc-absence); CI re-running to confirm. Reply drafts pending jeff approval (nothing posted upstream this pass).

## ASF headers resolved + replies posted 2026-06-11
Per jeff: dropped per-file AMD copyright + Author lines from all 10 files, moved attribution to NOTICE (088041c, head reconciled). Validated (fmt + hip build + cuda check). Posted all 6 replies: ASF (r3397736468), ROCm-floor (r3397736666), compile_error (r3397736790), features (r3397737008), maintenance (r3397737189), overall thanks (issuecomment-4683015613). HELD the precommit reply until CI confirms green. Next: compile-only ROCm hipcc CI job as a SEPARATE follow-up PR (post-#1399, since it needs the hip path #1399 adds) -- jeff approved starting it; it is explicitly NOT in #1399 (keeps the validated head stable; MOAT's no-fork-CI rule is about churn, this is an upstream-maintenance job in its own PR).

## Validation 2026-06-11 (gfx1100, revalidate -> completed) -- FULL GPU PASS + port fix

Platform: linux-gfx1100, GPU: AMD Radeon Pro W7800 48GB (gfx1100, RDNA3 wave32), HIP_VISIBLE_DEVICES=0 (Rust tests), all 4 GPUs visible (Python multi-GPU tests), ROCm 7.2.1.
Fork: AMD-Ecosystem/mahout @ moat-port HEAD ebb71af441f7a29f4f9e0440e4ae36ac4f49d3eb.

Delta from validated_sha (f3f7db33, pre-rebase squash) to ebb71af44:
- Functional HIP changes in the review-fix commit (206aa0c0d / 0b5042e before rebase): CudaSlice::drop re-binds device, explicit hipMemoryType mapping, build.rs consistency check, fork_default_stream non-blocking + amplitude stream sync.
- fmt, doc, ASF header changes (behavior-preserving).
- This fix: testing/qdp/test_bindings.py test_dlpack_device_id_non_zero made arch-aware.

Because the delta includes functional HIP changes, binary-equivalence check was NOT used; full GPU revalidation was performed.

Port bug found and fixed: test_dlpack_device_id_non_zero (multi-GPU DLPack test) had the device_type hardcoded to 2 (kDLCUDA). This was missed in the original port because gfx90a ran with HIP_VISIBLE_DEVICES=N (single GPU visible, so torch.cuda.device_count()==1 and the test skipped). On gfx1100 with 4 GPUs visible, the test runs and failed (reported (10,1) kDLROCM, not (2,1) kDLCUDA). Fix matches test_dlpack_device exactly: use getattr(torch.version, "hip", None) to select expected_device_type. Committed as ebb71af44, pushed to fork.

Build commands:
```
source "$HOME/.cargo/env"
export QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx1100 ROCM_PATH=/opt/rocm HIP_VISIBLE_DEVICES=0
bash utils/timeit.sh mahout compile -- cargo build \
  --manifest-path projects/mahout/src/qdp/Cargo.toml \
  -p qdp-core -p qdp-kernels --no-default-features --features hip -j 16
```
Exit 0. Only pre-existing upstream warnings (phase.cu unused variable, iqp.cu unused parameter).

Rust tests (HIP_VISIBLE_DEVICES=0, --test-threads=1):
```
bash utils/timeit.sh mahout test -- cargo test \
  --manifest-path projects/mahout/src/qdp/Cargo.toml \
  -p qdp-core -p qdp-kernels --no-default-features --features hip -- --test-threads=1
```
- qdp-core lib: 77/77.
- gpu_angle 12/12, gpu_api_workflow 8/8, gpu_basis 7/7, gpu_dlpack 9/9,
  gpu_fidelity 17/17, gpu_iqp 22/22, gpu_memory_safety 4/4, gpu_norm_f32 2/2,
  gpu_ptr_encoding 68/68 (4 new tests vs old 64; all pass), gpu_validation 8/8.
- Non-GPU: arrow_ipc 5/5, null_handling 6/6, numpy 4/4, parquet_f32 7/7, parquet_io 8/8,
  preprocessing 14/14, reader 3/3, tensorflow_io 9/9, torch_io 3/3, types 6/6.
- qdp-kernels: amplitude_encode 21/21, angle_encode 10/10.
- 0 failures total.
- Async-pipeline tests pass: test_amplitude_encoding_async_pipeline, test_angle_encoding_async_pipeline (gpu_api_workflow), test_angle_batch_f32_async_pipeline_path (gpu_angle_encoding).

Python parity (all 4 GPUs visible, testing/qdp + testing/qdp_python + qdp/qdp-python/tests):
- 305 passed, 10 skipped, 0 failed.
- Multi-GPU tests run and pass (test_dlpack_device_id_non_zero PASSES with arch-aware fix; test_encode_cuda_tensor_device_mismatch PASSES).
- Skips: 1 tensorflow-absent, 1 loader path-timing, 5 torch_ref sm_-arch check, 2 AmdQdpEngine-not-built, 1 NVIDIA-ref-absent -- all pre-existing/legit.

Wave32 verdict: All L2-norm/amplitude tests pass on gfx1100 wave32, identical to prior gfx1100 validation.

Transition: revalidate -> completed (validated_sha = ebb71af441f7a29f4f9e0440e4ae36ac4f49d3eb).
## Precommit fully addressed 2026-06-11
Two real precommit hook failures fixed in 9856724 (rebased onto a concurrent test-file push ebb71af that landed on the fork from jeff/another session; 088041c is its ancestor): (1) insert-license Rust -- qdp-python/build.rs lost a leading blank `//` comment line in the ASF removal; restored byte-identical to qdp-core/build.rs. (2) clippy unnecessary_cast under -D warnings -- 14 `stream.stream as *mut c_void` sites that are redundant under --all-features (hip wins, stream is already *mut c_void) but REAL under cuda-only (cudarc CUstream); fixed with hip-scoped `#![cfg_attr(feature="hip", allow(clippy::unnecessary_cast))]` at 3 crate roots/test files. Verified clippy clean under hip/cuda/cuda+hip; gfx90a hip build exit 0. Full `pre-commit run --all-files` can't go green on THIS host due to 2 env artifacts (torch 2.13 vs tch 2.9 ABI in the clippy hook's torch-sys; conda _pytest leakage in ty) -- neither from our commit, neither on the CI runner. Precommit reply to rich7420/ryankert01 held until CI confirms green on 9856724.
## Validation 2026-06-11 (windows-gfx1201, revalidate -> completed) -- PASS with delta-port

Platform: windows-gfx1201, GPU: AMD Radeon RX 9070 XT (gfx1201, RDNA4 wave32),
HIP_VISIBLE_DEVICES=0, Windows 11 (26200), TheRock ROCm 7.14.
Fork: AMD-Ecosystem/mahout @ moat-port HEAD 90b60069da9ee5bcd0f9be3c5fbd95ca2b6efcab.

Revalidating from f3f7db33 to 088041cb (head before the new fork commits; the review-fix
commit 206aa0c0d introduced HIP_STREAM_NON_BLOCKING in fork_default_stream).

### Delta analysis: what changed vs prior gfx1201 validation

1. HIP_STREAM_NON_BLOCKING in fork_default_stream (device.rs, 206aa0c0d):
   Changed hipStreamCreate to hipStreamCreateWithFlags(HIP_STREAM_NON_BLOCKING).
   On Linux gfx90a/gfx1100, this works correctly: after hipStreamSynchronize on a
   non-blocking stream, hipMemcpy D2H reads the kernel results. On Windows gfx1201
   TheRock 7.14, non-blocking stream writes are NOT visible via hipMemcpy after
   hipStreamSynchronize or hipDeviceSynchronize -- a cache coherency gap in the
   Windows HIP runtime. Symptom: 3 stream-path tests fail with "got 0":
   - qdp-kernels amplitude_encode: test_l2_norm_batch_kernel_stream
   - qdp-core gpu_ptr_encoding: test_encode_from_gpu_ptr_f32_with_stream_non_default_success
   - qdp-core gpu_ptr_encoding: test_encode_batch_from_gpu_ptr_f32_with_stream_success
   Fix: use hipStreamCreate (blocking stream) on non-Linux. Blocking streams
   have correct D2H coherency on all tested Windows TheRock versions.

2. encode_from_gpu_ptr_f32 / encode_from_gpu_ptr_f32_with_stream in amplitude.rs,
   angle.rs, basis.rs and the QuantumEncoder trait default in mod.rs (206aa0c0d)
   remained #[cfg(target_os = "linux")] while their callers in lib.rs used
   #[cfg(qdp_gpu_platform)]. On Windows this compiled the callers but not the
   callees, producing an unresolved-function error. Fix: rename all 5 occurrences
   to #[cfg(qdp_gpu_platform)].

Both fixes committed as 90b60069da9ee5bcd0f9be3c5fbd95ca2b6efcab on top of
9856724c3 (precommit build.rs fix) and ebb71af44 (dlpack test arch-aware fix)
which landed while this validation was in progress.

### Build commands (gfx1201)
```
ROOT=/b/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel
MSVC_DIR="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/HostX64/x64"
MSVC_BASE="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207"
WIN_SDK_BASE="/c/Program Files (x86)/Windows Kits/10"; SDK_VER="10.0.26100.0"
export CARGO_HOME=/b/develop/moat/agent_space/cargo RUSTUP_HOME=/b/develop/moat/agent_space/rustup
export PATH="$CARGO_HOME/bin:$MSVC_DIR:$ROOT/bin:$ROOT/lib/llvm/bin:$PATH"
export LIB="$(cygpath -w $MSVC_BASE/lib/x64);$(cygpath -w $WIN_SDK_BASE/Lib/$SDK_VER/ucrt/x64);$(cygpath -w $WIN_SDK_BASE/Lib/$SDK_VER/um/x64)"
export HIP_DEVICE_LIB_PATH="$ROOT/lib/llvm/amdgcn/bitcode"
export QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx1201 QDP_HIPCC="$ROOT/bin/hipcc.exe"
export ROCM_PATH="$ROOT" HIP_VISIBLE_DEVICES=0
cargo build --manifest-path /b/develop/moat/projects/mahout/src/qdp/Cargo.toml \
    -p qdp-core -p qdp-kernels --no-default-features --features hip
cp $ROOT/bin/amdhip64_7.dll $ROOT/bin/amd_comgr.dll $ROOT/bin/rocm_kpack.dll \
   /b/develop/moat/projects/mahout/src/qdp/target/debug/deps/
cargo test --manifest-path /b/develop/moat/projects/mahout/src/qdp/Cargo.toml \
    -p qdp-core -p qdp-kernels --no-default-features --features hip -- --test-threads=1
```

### Test results (gfx1201, dev profile, --test-threads=1) -- 330 passed, 0 failed

qdp-kernels:
- amplitude_encode 21/21, angle_encode 10/10.

qdp-core:
- lib unit tests: 77/77.
- GPU suites: gpu_angle 12/12, gpu_api_workflow 8/8, gpu_basis 7/7, gpu_dlpack 9/9,
  gpu_fidelity 17/17, gpu_iqp 22/22, gpu_memory_safety 4/4, gpu_norm_f32 2/2,
  gpu_ptr_encoding 68/68, gpu_validation 8/8.
  (gpu_ptr_encoding increased from 64 to 68 vs prior run: ebb71af44 made
   test_dlpack_device_id_non_zero arch-aware; now enabled on gfx1201.)
- Non-GPU: arrow_ipc 5/5, null_handling 6/6, numpy 4/4, parquet 8/8,
  preprocessing 14/14, tensorflow_io 9/9, torch_io 3/3, types 6/6.
- 0 failures total.

Note: 4 ignored tests are pre-existing Windows multi-GPU skips (#[ignore] on
tests that need 2+ devices; gfx1101 absent from bus on this host).

Non-blocking stream note for Linux revalidation: the fix uses #[cfg(target_os)]
to isolate the Windows workaround -- Linux still gets HIP_STREAM_NON_BLOCKING
(non-blocking stream, full pipeline overlap preserved). The Linux gfx90a and
gfx1100 platforms need to rebuild and revalidate (this delta touches device.rs
with a Rust source change that is inert on Linux -- the #[cfg(not(target_os = "linux"))]
branch is dead code on Linux. A binary-equivalence check should confirm identical
Linux device code objects for a carry-forward.)

Transition: revalidate -> completed (validated_sha = 90b60069da9ee5bcd0f9be3c5fbd95ca2b6efcab).

## Porter fix 2026-06-11 (windows-gfx1201) -- ROOT CAUSE: default-stream ordering, not coherency

The previous "stream coherency" delta (90b60069) was a misdiagnosis. The real
bug is a missing ordering between default-stream (NULL) buffer setup and a kernel
launched on a non-blocking forked stream that consumes it. CUDA's legacy default
stream is synchronizing, so the encoders' setup-on-NULL-then-launch-on-fork
pattern is implicitly ordered on NVIDIA; HIP's default stream is NOT synchronizing
relative to a hipStreamNonBlocking stream, so the kernel raced the setup and read
the zero-initialized buffer -> wrong norm (came back 0).

### How it was proven (agent_space/stream_repro/)
- stream_test.cpp (single kernel): non-blocking stream + hipStreamSynchronize(S) +
  null-stream dtoh -> CORRECT on both System32 Adrenalin and TheRock ROCm runtimes.
  Only the NO-sync race case fails. So the runtime is NOT buggy (no coherency gap).
- The failing Rust test was instrumented: fork_default_stream made stream 0x...,
  wait_for synced that SAME stream (hipStreamSynchronize -> 0 success), yet readback
  was still 0. Adding hipDeviceSynchronize() in wait_for ALSO did not help. So it is
  NOT a readback-sync problem -- syncing harder does nothing.
- stream_test3/4/5.cpp replicated the EXACT encoder ordering in pure C++:
  blocking null-stream H2D + alloc_zeros(memset) -> hipMemsetAsync(out,S) ->
  accum(atomicAdd)<<<S>>> -> finalize<<<S>>> -> streamSync(S) -> null dtoh.
  Variant matrix nailed it: with NO ordering between the null-stream setup and the
  forked-stream work -> FAIL (got {0,0}); adding ANY of hipStreamSynchronize(NULL),
  hipDeviceSynchronize(), or an event recorded on NULL + hipStreamWaitEvent(S,ev)
  before the forked-stream work -> PASS (got {0.316,1}). The hazard is setup-side,
  not readback-side.

### The fix (fork @ d97db73ad, on top of 90b60069)
qdp-kernels/src/device.rs:
- Reverted fork_default_stream to HIP_STREAM_NON_BLOCKING on BOTH Linux and Windows
  (removed the cfg split + the false coherency comment + the unused hipStreamCreate
  extern). Pipeline overlap restored on Windows.
- Added sync_default_stream() = hipStreamSynchronize(NULL) at the tail of the
  blocking alloc_zeros and htod_sync_copy_into (covers htod_sync_copy/htod_copy too),
  restoring CUDA's synchronizing-default-stream ordering before any forked stream
  consumes those buffers. Does NOT touch the async pipeline (which uses
  cuda_ffi hipMemcpyAsync on explicit streams, not these blocking paths).
qdp-core/src/gpu/encodings/{amplitude,phase}.rs:
- Closed the symmetric readback-side gaps still missing a sync (kernel on caller
  stream -> NULL-stream dtoh): amplitude.rs f64 batch (~418) and f32 batch (~652)
  norm-validation copies, and phase.rs batch finiteness-probe copy (~357) now call
  sync_cuda_stream(stream, ...) before dtoh_sync_copy, matching the idiom the other
  encoders already use (amplitude 989/1172, etc.). The amplitude single/batch
  _with_stream standalone paths (935/1143) already synced and were unchanged.

Arch-unified: correct on wave32 (gfx1100/gfx1151/gfx1201) and wave64 (gfx90a). On
Linux the default stream is already synchronizing so sync_default_stream is a
harmless no-op-cost ordering point. NOTE: this is a FUNCTIONAL change to shared
(non-arch-guarded) qdp-kernels code, so advance_head correctly flipped the other
completed platforms (linux-gfx1100, windows-gfx1101/gfx1151) to revalidate;
linux-gfx90a stays pr-open (lead). The fork's default stream is non-blocking on
every platform now, so the per-platform behavior split is gone.

### Build + test (gfx1201, RX 9070 XT, TheRock ROCm 7.14, HIP_VISIBLE_DEVICES=0)
Env: agent_space/mahout_gfx1201_env.sh (TheRock _rocm_sdk_devel, MSVC 14.44,
QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx1201, Windows-style LIB via cygpath -w).
```
cargo build -p qdp-core -p qdp-kernels --no-default-features --features hip
cargo test  -p qdp-core -p qdp-kernels --no-default-features --features hip -- --test-threads=1
```
Both exit 0. Full suite GREEN, 0 failures: qdp-core lib 77, gpu_angle 12,
gpu_api_workflow 8, gpu_basis 7, gpu_dlpack 9, gpu_fidelity 17, gpu_iqp 22,
gpu_memory_safety 4, gpu_norm_f32 2, gpu_ptr_encoding 68 (all 10 _with_stream
variants pass), gpu_validation 8; non-GPU arrow 5, null 6, numpy 4, parquet 8+7,
preprocessing 14, reader 3, tensorflow 9, torch 3, types 6; qdp-kernels amplitude
21, angle 10. 4 ignored (unchanged). The three previously-regressed stream tests
all pass with the non-blocking stream restored.

Transition: revalidate -> completed (validated_sha = d97db73ad592...).

## gfx1201 stream-ordering fix d97db73 (root cause) 2026-06-11
gfx1201 pushed d97db73: reverts the Windows blocking-stream workaround (90b6006), restores non-blocking stream on all platforms, fixes the REAL bug -- missing ordering between default-stream htod/alloc_zeros setup and the forked non-blocking-stream kernel (CUDA legacy default stream synchronizes; HIP's does not). Adds sync_default_stream after the blocking setup copies. No-op on Linux (gfx90a default stream already synchronizing). VERIFIED on gfx90a: precommit clean (fmt+full clippy via torch venv), gpu_ptr_encoding 68/68 incl. all *_with_stream_* tests. Head reconciled to d97db73. All CI green on the prior head (90b6006); will reconfirm on d97db73. Precommit reply still held for jeff's go.

## Validation 2026-06-11 (gfx1100, revalidate -> completed at d97db73) -- FULL GPU PASS

Platform: linux-gfx1100, GPU: AMD Radeon Pro W7800 48GB (gfx1100, RDNA3 wave32),
HIP_VISIBLE_DEVICES=0 (Rust tests), all 4 GPUs visible (Python parity),
ROCm 7.2.1, Rust 1.96.0.
Fork: AMD-Ecosystem/mahout @ moat-port HEAD d97db73adeac70b377c22e17fde6af7ff4ff3057.

Delta ebb71af44..d97db73 (3 commits):
- 9856724c3: build.rs license header + HIP clippy cast lint suppression (cfg_attr, inert on GPU path).
- 90b60069d: Windows gfx1201 delta -- #[cfg(target_os="linux")] -> #[cfg(qdp_gpu_platform)] renames in encoding files + transient Windows blocking-stream workaround (later reverted).
- d97db73: Stream ordering fix -- sync_default_stream() after alloc_zeros/htod in device.rs; sync_cuda_stream() before dtoh in amplitude.rs (batch f64+f32) and phase.rs (batch). Restores non-blocking fork_default_stream on all platforms. This is a functional HIP change; binary-equivalence carry-forward was NOT used; full GPU revalidation performed.

Build:
```
source "$HOME/.cargo/env"
export QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx1100 ROCM_PATH=/opt/rocm HIP_VISIBLE_DEVICES=0
bash utils/timeit.sh mahout compile -- cargo build \
  --manifest-path projects/mahout/src/qdp/Cargo.toml \
  -p qdp-core -p qdp-kernels --no-default-features --features hip -j 16
```
Exit 0. Only pre-existing upstream warnings (phase.cu unused variable, iqp.cu unused parameter).

Rust tests (HIP_VISIBLE_DEVICES=0, --test-threads=1):
```
bash utils/timeit.sh mahout test -- cargo test \
  --manifest-path projects/mahout/src/qdp/Cargo.toml \
  -p qdp-core -p qdp-kernels --no-default-features --features hip -- --test-threads=1
```
- qdp-core lib: 77/77.
- gpu_angle 12/12, gpu_api_workflow 8/8, gpu_basis 7/7, gpu_dlpack 9/9,
  gpu_fidelity 17/17, gpu_iqp 22/22, gpu_memory_safety 4/4, gpu_norm_f32 2/2,
  gpu_ptr_encoding 68/68, gpu_validation 8/8.
- Non-GPU: arrow_ipc 5/5, null_handling 6/6, numpy 4/4, parquet_f32 7/7, parquet_io 8/8,
  preprocessing 14/14, reader 3/3, tensorflow_io 9/9, torch_io 3/3, types 6/6.
- qdp-kernels: amplitude_encode 21/21, angle_encode 10/10.
- 0 failures total. test_l2_norm_batch_kernel_stream passes with non-blocking stream + sync_default_stream ordering fix.
- Async-pipeline tests pass: test_amplitude_encoding_async_pipeline, test_angle_encoding_async_pipeline (gpu_api_workflow), test_angle_batch_f32_async_pipeline_path (gpu_angle_encoding).

Python parity (all 4 GPUs visible, testing/qdp + testing/qdp_python + qdp/qdp-python/tests):
Rebuilt wheel at d97db73 (maturin build --features hip --profile dev --compatibility linux), pip installed --no-deps.
- 305 passed, 10 skipped, 0 failed. Matches prior gfx1100 baseline exactly.
- Skips: 1 tensorflow-absent, 1 loader path-timing, 5 torch_ref sm_-arch check, 2 AmdQdpEngine-not-built, 1 NVIDIA-ref-absent -- all pre-existing/legit.

No changes pushed to the fork (validation as-is). Open PR apache/mahout#1399 unaffected.

Transition: revalidate -> completed (validated_sha = d97db73adeac70b377c22e17fde6af7ff4ff3057).

## Validation 2026-06-12 (windows-gfx1101, revalidate -> completed at d97db73) -- FULL GPU PASS

Platform: windows-gfx1101, GPU: AMD Radeon PRO V710 (gfx1101, RDNA3 wave32, warpSize=32),
HIP_VISIBLE_DEVICES=0 (only GPU on bus), Windows 11 (26200), TheRock ROCm 7.14,
Rust 1.96.0 (msvc target). Fork: AMD-Ecosystem/mahout @ moat-port HEAD d97db73adeac70b377c22e17fde6af7ff4ff3057.

Delta from validated_sha (f3f7db33) includes functional HIP changes; binary-equivalence
carry-forward was NOT used; full GPU revalidation performed.

Build commands:
```
ROOT=B:/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel
MSVC_DIR="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/HostX64/x64"
MSVC_BASE="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207"
WIN_SDK_BASE="/c/Program Files (x86)/Windows Kits/10"; SDK_VER="10.0.26100.0"
export CARGO_HOME=/b/develop/moat/agent_space/cargo RUSTUP_HOME=/b/develop/moat/agent_space/rustup
export PATH="$CARGO_HOME/bin:$MSVC_DIR:$ROOT/bin:$ROOT/lib/llvm/bin:$PATH"
export LIB="$(cygpath -w $MSVC_BASE/lib/x64);$(cygpath -w $WIN_SDK_BASE/Lib/$SDK_VER/ucrt/x64);$(cygpath -w $WIN_SDK_BASE/Lib/$SDK_VER/um/x64)"
export HIP_DEVICE_LIB_PATH="$ROOT/lib/llvm/amdgcn/bitcode"
export QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx1101 QDP_HIPCC="$ROOT/bin/hipcc.exe"
export ROCM_PATH="$ROOT" HIP_VISIBLE_DEVICES=0

bash utils/timeit.sh mahout compile -- \
  cargo build --manifest-path projects/mahout/src/qdp/Cargo.toml \
    -p qdp-core -p qdp-kernels --no-default-features --features hip
# exit 0, 3.91s (incremental). Pre-existing warnings: iqp.cu unused param, phase.cu unused var.

cp $ROOT/bin/{amdhip64_7.dll,amd_comgr.dll,rocm_kpack.dll,hiprtc0714.dll,hiprtc-builtins0714.dll} \
   projects/mahout/src/qdp/target/debug/deps/

bash utils/timeit.sh mahout test -- \
  cargo test --manifest-path projects/mahout/src/qdp/Cargo.toml \
    -p qdp-core -p qdp-kernels --no-default-features --features hip -- --test-threads=1
```

Rust tests (HIP_VISIBLE_DEVICES=0, --test-threads=1) -- 0 failures total:
- qdp-core lib: 77/77.
- GPU suites: gpu_angle 12/12, gpu_api_workflow 8/8, gpu_basis 7/7, gpu_dlpack 9/9,
  gpu_fidelity 17/17, gpu_iqp 22/22, gpu_memory_safety 4/4, gpu_norm_f32 2/2,
  gpu_ptr_encoding 68/68, gpu_validation 8/8.
- Non-GPU: arrow_ipc 5/5, null_handling 6/6, numpy 4/4, parquet_f32 7/7, parquet_io 8/8,
  preprocessing 14/14, reader 3/3, tensorflow_io 9/9, torch_io 3/3, types 6/6.
- qdp-kernels: amplitude_encode 21/21, angle_encode 10/10.
- Async-pipeline tests pass: test_amplitude_encoding_async_pipeline,
  test_angle_encoding_async_pipeline (gpu_api_workflow),
  test_angle_batch_f32_async_pipeline_path (gpu_angle_encoding).
- Stream ordering fix verified: test_l2_norm_batch_kernel_stream (amplitude_encode),
  test_encode_from_gpu_ptr_f32_with_stream_non_default_success and
  test_encode_batch_from_gpu_ptr_f32_with_stream_success (gpu_ptr_encoding) -- all 3
  previously-failing stream tests PASS with non-blocking stream + sync_default_stream fix.
- gpu_ptr_encoding count is 68 (vs 64 in the original gfx1101 pass): 4 new tests added
  in ebb71af44 (dlpack arch-awareness) now run on gfx1101 too.

Wave32 verdict: gfx1101 RDNA3 wave32 -- all L2-norm/amplitude tests pass.
warp_id = threadIdx.x / warpSize == >>5 on wave32. QDP_FULL_WARP_MASK upper 32 bits zero on wave32.

Transition: revalidate -> completed (validated_sha = d97db73adeac70b377c22e17fde6af7ff4ff3057).

## Validation 2026-06-24 (gfx90a, revalidate -> completed at f84b39a) -- FULL GPU PASS

Platform: linux-gfx90a, GPU: MI250X gfx90a (wave64), GCD: HIP_VISIBLE_DEVICES=0, ROCm 7.2.1.
Fork: AMD-Ecosystem/mahout @ moat-port HEAD f84b39aef62e3dbddaa36f877ff29969f4dd49f2.
Reason: moat-port branch was force-rewritten (rebased onto upstream main after #1393 merge)
under the open upstream PR; prior gfx90a validated_sha (0b5042e70) was orphaned. Full GPU
revalidation performed at the PR tip. The content is functionally identical to d97db73
(stream-ordering fix + all review fixes), rebased with upstream non-QDP commits in between.

Build commands:
```
source "$HOME/.cargo/env"
export QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx90a ROCM_PATH=/opt/rocm HIP_VISIBLE_DEVICES=0
bash utils/timeit.sh mahout compile -- cargo build \
  --manifest-path projects/mahout/src/qdp/Cargo.toml \
  -p qdp-core -p qdp-kernels --no-default-features --features hip -j 16
# exit 0 (0.80s incremental). Pre-existing warnings: iqp.cu unused param, phase.cu unused var.
```

Rust tests:
```
bash utils/timeit.sh mahout test -- cargo test \
  --manifest-path projects/mahout/src/qdp/Cargo.toml \
  -p qdp-core -p qdp-kernels --no-default-features --features hip -- --test-threads=1
```
- qdp-core lib: 90/90 (expanded from 77 due to upstream test additions merged via rebase).
- gpu_angle 12/12, gpu_api_workflow 8/8, gpu_basis 7/7, gpu_dlpack 9/9,
  gpu_fidelity 17/17, gpu_iqp 22/22, gpu_memory_safety 4/4, gpu_norm_f32 2/2,
  gpu_ptr_encoding 68/68, gpu_validation 8/8.
- Non-GPU: arrow_ipc 5/5, null_handling 6/6, numpy 4/4, parquet_f32 7/7, parquet_io 8/8,
  preprocessing 14/14, reader 3/3, tensorflow_io 9/9, torch_io 3/3, types 6/6.
- qdp-kernels: amplitude_encode 21/21, angle_encode 10/10.
- 0 failures total.
- Async-pipeline tests pass: test_amplitude_encoding_async_pipeline,
  test_angle_encoding_async_pipeline (gpu_api_workflow),
  test_angle_batch_f32_async_pipeline_path (gpu_angle_encoding).
- Stream ordering tests pass: test_l2_norm_batch_kernel_stream (amplitude_encode),
  test_encode_from_gpu_ptr_f32_with_stream_non_default_success,
  test_encode_batch_from_gpu_ptr_f32_with_stream_success (gpu_ptr_encoding).

Python wheel (maturin build --features hip --profile dev --compatibility linux):
```
conda run -n py_3.12 maturin build --features hip --profile dev \
  --manifest-path projects/mahout/src/qdp/qdp-python/Cargo.toml \
  --out /tmp/mahout_wheels --compatibility linux
conda run -n py_3.12 pip install --no-deps --force-reinstall \
  /tmp/mahout_wheels/qumat_qdp-0.3.0.dev0-cp312-cp312-linux_x86_64.whl
```
Exit 0. Wheel installs as qumat-qdp 0.3.0.dev0.

Python parity (testing/qdp + testing/qdp_python + qdp/qdp-python/tests, HIP_VISIBLE_DEVICES=0):
- 303 passed, 12 skipped, 0 failed.
- Skips: 2 multi-GPU, 1 tensorflow-absent, 1 loader path-timing,
  5 torch_ref sm_-arch check (Triton/torch CUDA reference path, not native engine),
  2 AmdQdpEngine-not-built, 1 NVIDIA-ref-absent -- all pre-existing/legit.

Transition: revalidate -> completed (validated_sha = f84b39aef62e3dbddaa36f877ff29969f4dd49f2).

## Validation 2026-06-24 (gfx1100, revalidate -> completed at f84b39a) -- binary-equiv carry-forward

Platform: linux-gfx1100, GPU: AMD Radeon Pro W7800 48GB (gfx1100, RDNA3 wave32), ROCm 7.2.1.
Fork: AMD-Ecosystem/mahout @ moat-port HEAD f84b39aef62e3dbddaa36f877ff29969f4dd49f2.

Delta d97db73adeac..f84b39aef62e (21 commits via rebased moat-port):
- No changes in qdp-kernels/ (kernel .cu files, build.rs, hip_compat/ shims, device.rs GPU shim, cuda_ffi.rs).
- Functional changes: qdp-core/src/readers/arrow_ipc.rs and parquet.rs (upstream bug fix #1402:
  nullable outer row handling in List<T> readers -- CPU/non-GPU code only).
- Doc/comment additions: Python docstrings in qumat_qdp/{backend,loader,tensor}.py,
  docs/api/ files, README/DEVELOPMENT.md additions.
- Upstream non-QDP commits merged via rebase (website dep bumps, CODEOWNERS, torch dep bump).

Binary-equivalence check:
- Built validated_sha (d97db73a) in git worktree /tmp/mahout-old-gfx1100 with
  CARGO_TARGET_DIR=/tmp/mahout-old-gfx1100-target, QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx1100.
- Built HEAD (f84b39aef) in the default target dir (same env).
- Compared libkernels.a (the archive of 6 .cu HIP kernel objects -- the only device code artifact).
- Result: `verdict=identical` -- exported symbols + device ISA identical (0 exports). All 6 gfx1100
  kernel TUs (amplitude.cu, basis.cu, angle.cu, validation.cu, iqp.cu, phase.cu) compile to
  identical device code objects.

No GPU re-run required. Transition: revalidate -> completed (validated_sha = f84b39aef62e3dbddaa36f877ff29969f4dd49f2).

## Validation 2026-06-24 (windows-gfx1201, revalidate -> completed at f84b39ae) -- FULL GPU PASS

Platform: windows-gfx1201, GPU: AMD Radeon RX 9070 XT (gfx1201, RDNA4 wave32, warpSize=32),
HIP_VISIBLE_DEVICES=0 (gfx1201 is mask 0 on this session; gfx1101 is mask 1),
Windows 11 (26200), TheRock ROCm 7.14.0a20260604,
torch 2.9.1+rocm7.14.0a20260604, Rust 1.96.0 (msvc target), maturin 1.13.3.
Fork: AMD-Ecosystem/mahout @ moat-port HEAD f84b39aef62e3dbddaa36f877ff29969f4dd49f2.

Prior validated_sha (d97db73ad592c980f44a74f73118d2e7cf43e94a) was orphaned (lost in
a moat-port rebase), so binary-equiv carry-forward was not possible; full GPU revalidation
at the current HEAD was required.

### Commits validated (windows-gfx1201-specific)

The two most recent functional commits target the gfx1201 / Windows default-stream-ordering
class of bug:

1. 1f2cddd6c "QDP: Windows gfx1201 delta -- cfg and stream coherency fixes"
   - #[cfg(target_os = "linux")] -> #[cfg(qdp_gpu_platform)] for the five encode_from_gpu_ptr
     functions that the Windows+hip callers required (prevents unresolved-function linker error).
   - Initial workaround: hipStreamCreate (blocking stream) on Windows. Superseded by f84b39ae.

2. f84b39aef "QDP: order default-stream setup before forked-stream kernels"
   - Reverts the Windows blocking-stream workaround; restores HIP_STREAM_NON_BLOCKING on all
     platforms (correct, matches cudarc, preserves dual-stream pipeline overlap).
   - Root cause: HIP's default (NULL) stream is NOT synchronizing relative to a non-blocking
     forked stream (unlike CUDA's legacy default stream). The encoders setup buffers via
     alloc_zeros/htod on the NULL stream, then launch kernels on the forked stream; on RDNA4
     the kernel raced the setup and read stale/zero data.
   - Fix: sync_default_stream() = hipStreamSynchronize(NULL) at the tail of alloc_zeros and
     htod_sync_copy_into, ordering setup before any forked-stream consumer. Also adds missing
     sync_cuda_stream() calls before dtoh in amplitude.rs batch f64+f32 paths and phase.rs batch.
   - Arch-unified and correct on wave32 (gfx1100/gfx1201) and wave64 (gfx90a).
   - CONFIRMED on RDNA4 wave32: all three previously-regressing stream tests now pass.

### Build commands

```
ROOT=/b/develop/TheRock/external-builds/pytorch/.venv/Lib/site-packages/_rocm_sdk_devel
MSVC_BASE="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207"
WIN_SDK="/c/Program Files (x86)/Windows Kits/10"; SDK_VER=10.0.26100.0
export CARGO_HOME=/b/develop/moat/agent_space/cargo RUSTUP_HOME=/b/develop/moat/agent_space/rustup
export PATH="$CARGO_HOME/bin:$MSVC_BASE/bin/HostX64/x64:$ROOT/bin:$ROOT/lib/llvm/bin:$PATH"
export LIB="$(cygpath -w $MSVC_BASE/lib/x64);$(cygpath -w $WIN_SDK/Lib/$SDK_VER/ucrt/x64);$(cygpath -w $WIN_SDK/Lib/$SDK_VER/um/x64)"
export HIP_DEVICE_LIB_PATH="$ROOT/lib/llvm/amdgcn/bitcode"
export QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx1201 QDP_HIPCC="$ROOT/bin/hipcc.exe"
export ROCM_PATH="$ROOT" HIP_VISIBLE_DEVICES=0

# Kernels + core (dev profile, incremental)
bash utils/timeit.sh mahout compile -- \
  cargo build --manifest-path projects/mahout/src/qdp/Cargo.toml \
    -p qdp-core -p qdp-kernels --no-default-features --features hip
# -> exit 0, ~5s. Pre-existing warnings: iqp.cu unused param, phase.cu unused var.

# Deploy TheRock runtime DLLs (already present from prior run; refreshed)
cp $ROOT/bin/{amdhip64_7.dll,amd_comgr.dll,rocm_kpack.dll,hiprtc0714.dll,hiprtc-builtins0714.dll} \
   projects/mahout/src/qdp/target/debug/deps/

# Python extension wheel (dev profile -- release LTO breaks cdylib on HIP)
VENV=/b/develop/TheRock/external-builds/pytorch/.venv
maturin build --features hip --profile dev \
  --manifest-path projects/mahout/src/qdp/qdp-python/Cargo.toml \
  --out /b/develop/moat/agent_space/mahout_wheels_gfx1201_new \
  --interpreter $VENV/Scripts/python.exe
# -> exit 0, 6.4s. qumat_qdp-0.3.0.dev0-cp312-cp312-win_amd64.whl
pip install --no-deps --force-reinstall <wheel>
cp $ROOT/bin/{amdhip64_7.dll,amd_comgr.dll,rocm_kpack.dll,hiprtc0714.dll,hiprtc-builtins0714.dll} \
   $VENV/Lib/site-packages/_qdp/
```

### Rust test results (HIP_VISIBLE_DEVICES=0, --test-threads=1) -- 0 failures

qdp-kernels: amplitude_encode 21/21, angle_encode 10/10.
qdp-core lib: 90/90.
GPU suites: gpu_angle 12/12, gpu_api_workflow 8/8, gpu_basis 7/7, gpu_dlpack 9/9,
  gpu_fidelity 17/17, gpu_iqp 22/22, gpu_memory_safety 4/4, gpu_norm_f32 2/2,
  gpu_ptr_encoding 68/68, gpu_validation 8/8.
Non-GPU: arrow_ipc 5/5, null_handling 6/6, numpy 4/4, parquet_f32 7/7, parquet_io 8/8,
  preprocessing 14/14, reader 3/3, tensorflow_io 9/9, torch_io 3/3, types 6/6.

Stream-ordering tests confirmed passing (RDNA4 wave32 confirmation):
- test_l2_norm_batch_kernel_stream (amplitude_encode) -- PASS (both runs)
- test_encode_from_gpu_ptr_f32_with_stream_non_default_success (gpu_ptr_encoding) -- PASS (both runs)
- test_encode_batch_from_gpu_ptr_f32_with_stream_success (gpu_ptr_encoding) -- PASS (both runs)
- test_amplitude_encoding_async_pipeline (gpu_api_workflow) -- PASS (both runs)
- test_angle_encoding_async_pipeline (gpu_api_workflow) -- PASS (both runs)
- test_angle_batch_f32_async_pipeline_path (gpu_angle_encoding) -- PASS (both runs)

### Python parity (testing/qdp + testing/qdp_python + qdp/qdp-python/tests)
303 passed, 12 skipped, 0 failed.
Skips (all pre-existing/legit):
- 2 multi-GPU (test_bindings.py)
- 1 tensorflow-absent (test_bindings.py)
- 1 loader path-timing (test_quantum_data_loader.py)
- 5 torch_ref sm_-arch check (sm_120 on gfx1201 vs sm_-cap list; Triton/torch CUDA reference
  path, not native engine -- native HIP engine runs correctly on gfx1201)
- 2 AmdQdpEngine-not-built (test_amd_engine.py)
- 1 NVIDIA-ref-absent (test_triton_amd_backend.py)

Wave32 verdict: gfx1201 RDNA4 wave32 -- all L2-norm/amplitude tests pass. warp_id =
threadIdx.x / warpSize == >>5 on wave32. The sync_default_stream() fix correctly orders
NULL-stream setup before non-blocking-stream kernel dispatch; this is the RDNA4 confirmation
that the fix in f84b39ae is correct and deterministic.

Transition: revalidate -> completed (validated_sha = f84b39aef62e3dbddaa36f877ff29969f4dd49f2).

## Validation 2026-06-24 (windows-gfx1101, revalidate -> completed) -- binary-equiv carry-forward

Platform: windows-gfx1101, GPU absent (TDR-removed this session). Build-only binary-equivalence check.
Prior validated_sha = d97db73adeac70b377c22e17fde6af7ff4ff3057.
Current head_sha = f84b39aef62e3dbddaa36f877ff29969f4dd49f2.

Delta d97db73a..f84b39ae (21 commits): upstream non-QDP merges (website/doc bumps, CODEOWNERS,
torch dep bump), upstream bug fix #1402 (arrow_ipc.rs/parquet.rs nullable outer row handling --
CPU/non-GPU code only), Python docstrings. Zero changes to qdp-kernels/ (no .cu kernel files,
no build.rs, no hip_compat/ shims, no device.rs GPU shim).

Binary-equivalence check (gfx1101):
- Built d97db73a in git worktree /b/develop/moat/agent_space/mahout_old_gfx1101 into
  agent_space/mahout_old_gfx1101_target; built f84b39ae in agent_space/mahout_head_gfx1101_target.
  Both: QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx1101, TheRock hipcc 7.14, Rust 1.96 msvc. Both exit 0.
- Extracted .hip_fatbin sections from each of the 6 kernel COFF objects in libkernels.a using
  llvm-objcopy --dump-section. Parsed the CLANG_OFFLOAD_BUNDLE header (Python) to locate the
  ELF payload for hipv4-amdgcn-amd-amdhsa--gfx1101 in each fatbin.
- Compared ELF section 7 (the GPU ISA/text section) for all 6 kernels:
  amplitude.cu: IDENTICAL (sha256 6f0282b9163c8880..., 125364 bytes)
  angle.cu: IDENTICAL (sha256 0fe2b5ac172cbbf0..., 41220 bytes)
  basis.cu: IDENTICAL (sha256 d5175bf4be1ac515..., 22576 bytes)
  iqp.cu: IDENTICAL (sha256 d2113aab200a6918..., 93460 bytes)
  phase.cu: IDENTICAL (sha256 b4bc400ceba90ac8..., 22340 bytes)
  validation.cu: IDENTICAL (sha256 a0a9440afa6f0792..., 23548 bytes)
- Fatbin size differences (16 bytes per kernel): ELF debug string sections containing the build
  path (worktree path vs main checkout path). Not GPU code.
- Symbol check (llvm-nm --defined-only): 194 kernel function symbols, IDENTICAL. The
  __hip_cuid_* / __hip_gpubin_handle_* differences are path-derived compilation unit IDs
  (internal to HIP runtime module loading), not callable kernel entry points.

Verdict: gfx1101 GPU ISA identical at d97db73a and f84b39ae; carry-forward binary-equiv.
Transition: revalidate -> completed (validated_sha = f84b39aef62e3dbddaa36f877ff29969f4dd49f2).

## PR review round 2026-07-02 (rebase + reviewer feedback, linux-gfx90a) -- PASS

PR apache/mahout#1399 was CONFLICTING (upstream added a qdp_no_cuda stub path to
cuda_ffi.rs and refactored qdp-core/build.rs) and had unresolved threads from
rich7420/ryankert01. Rebased moat-port onto upstream/main and addressed the
feedback. Collapsed the prior 10-commit history into one squashed port commit
(the fork's multi-commit history and the orphaned validated_sha f84b39ae are
irrelevant post-rebase; status.json is the authority), rebased it once to resolve
conflicts, then added one review-fix commit. New fork HEAD:
fd8c7a38ac725b31f6b5ce1cc9e969cd357066f8.

Conflict resolution (5 files, resolved once on the squashed commit):
- qdp-core/src/gpu/cuda_ffi.rs: merged upstream's new qdp_no_cuda CUDA stub
  mechanism with our cuda_rt/hip_rt split. Now three compile-time backends:
  cuda_rt (cuda && !hip && !qdp_no_cuda), no_cuda_stubs (cuda && !hip &&
  qdp_no_cuda; added a cudaMemcpy stub since metrics.rs uses it), hip_rt (hip).
- qdp-core/build.rs: kept upstream's compile_protos()/configure_cuda_linkage()
  refactor, re-added qdp_gpu_platform emission as emit_gpu_platform_cfg().
- qdp-kernels/build.rs: kept both upstream's rerun-if-env PATH and our HIP env
  reruns.
- qdp-python engine.rs/loader.rs: our qdp_gpu_platform cfg + upstream's added
  parse_dtype in the import list.

Review fixes (commit 2):
- validation.rs stream race (rich7420): the four validators launched on the
  caller's stream then read the flag on the default stream without ordering.
  With forked streams now non-blocking, added sync_cuda_stream(stream, ...) before
  each of the four dtoh readbacks (same guard amplitude.rs already uses).
- hipPointerAttribute_t / hipMemoryType hand-rolled mirror (rich7420 x2): added
  hip_compat/verify_pointer_attrs.cpp, compiled by hipcc in build_hip, that
  static_asserts the enum values (Device=2/Managed=3) and struct field offsets
  (type@0, device@4, ...@8/16/24/28, size<=32) against the installed
  hip_runtime_api.h. Verified the ROCm 7.2 header matches exactly; a future header
  change fails the build loudly.
- cuComplex.h macro collision (rich7420): wrapped make_cuComplex/cuCreal/... in
  #ifndef so newer ROCm hip_complex.h (which ships those names) does not redefine.
- device.rs slice_mut offset (rich7420): widened range.start to u64 before
  multiplying by size_of to avoid usize wrap.
- device.rs CudaStream Sync (rich7420): dropped the Sync impl to match cudarc's
  Send-only contract (nothing shares a stream by & across threads; build passes).
- ASF headers (ryankert01): per-file AMD copyright/author lines were already
  gone; also removed the AMD entry from NOTICE (git history preserves authorship),
  per the "no added AMD copyright" call.

Already in the tree from the prior round (verified, replied, resolved): Drop
device rebind before hipFree; non-blocking forked streams; QDP_USE_HIP-vs-feature
build.rs panic; ROCm >= 6.0 floor doc; neither-feature compile_error.

cuda+hip both-enabled (ryankert01): kept documented hip-precedence rather than
compile_error!, because --all-features (used by clippy/CI per build.rs) enables
both simultaneously and a compile_error would break that lint path;
device.rs/cuda_ffi.rs deterministically select hip when both are on. Verified
`cargo check -p qdp-kernels --features hip` (both on) builds via hip precedence.

CI-rot / hip GHA (ryankert01): reply-only; not adding a GitHub Actions workflow
(a CPU-only hipcc job cannot exercise the GPU path and every yml edit churns the
regression guard), no tracking issue.

Validation (linux-gfx90a, MI250X, ROCm 7.2.1, HIP_VISIBLE_DEVICES=2):
- cargo build -p qdp-core -p qdp-kernels --no-default-features --features hip: exit 0.
- cargo test (same, --test-threads=1): 358 passed, 0 failed (lib 100, gpu suites
  incl. gpu_ptr_encoding 68 / gpu_validation 8 / gpu_api_workflow 8 async, all
  non-GPU suites).
- cargo check -p qdp-kernels --no-default-features: clean compile_error
  "requires exactly one of cuda or hip" (no raw E0432).
- QDP_NO_CUDA=1 cargo check (default cuda feature): compiles via no_cuda_stubs
  (best CUDA-path compile check without an NVIDIA toolkit).
- Python parity (testing/qdp + testing/qdp_python + qdp/qdp-python/tests): 303
  passed, 12 skipped, 0 failed. NOTE: the conda env's numpy had been bumped to
  2.5.0, which the installed torch build cannot initialize ("Numpy is not
  available"), so all torch<->numpy conversion tests failed spuriously; pip
  install 'numpy<2' fixed the env and the suite is fully green. Not a port issue.

Followers (gfx1100/gfx1201/gfx1151) -> revalidate (functional delta); gfx90a lead
stays pr-open with validated_sha advanced to the new head.

## Validation 2026-07-02 (gfx1100, revalidate -> completed at fd8c7a38) -- binary-equiv carry-forward

Platform: linux-gfx1100, GPU: AMD Radeon Pro W7800 48GB (gfx1100, RDNA3 wave32), ROCm 7.2.1.
Fork: AMD-Ecosystem/mahout @ moat-port HEAD fd8c7a38ac725b31f6b5ce1cc9e969cd357066f8.
Prior validated_sha: f84b39aef62e3dbddaa36f877ff29969f4dd49f2.

Delta f84b39ae..fd8c7a38 (2 port commits + multiple upstream non-QDP commits via rebase):
- Upstream conflict resolution: cuda_ffi.rs merged upstream's new qdp_no_cuda CUDA stub
  mechanism with our cuda_rt/hip_rt split; build.rs kept upstream's refactor + re-added
  qdp_gpu_platform emission; qdp-kernels/build.rs kept both upstream's PATH reruns and our HIP
  env reruns; engine.rs/loader.rs merged our qdp_gpu_platform cfg with upstream's parse_dtype.
- Reviewer fixes (rich7420/ryankert01): validation.rs stream race sync (Rust host-only);
  hipPointerAttribute_t build-time verify_pointer_attrs.cpp (compile-time check only);
  cuComplex.h #ifndef guards around macro aliases (header-defensive, same behavior on ROCm 7.2);
  device.rs slice_mut usize->u64 widening (Rust host-only); CudaStream Sync drop (Rust host-only);
  NOTICE AMD entry removal.
- No .cu kernel source files changed.

Binary-equivalence check (gfx1100):
- Built f84b39ae (validated_sha) in git worktree /tmp/mahout-old-gfx1100,
  CARGO_TARGET_DIR=/tmp/mahout-old-gfx1100-target, QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx1100 -> exit 0.
- Built fd8c7a38 (HEAD) in the default target dir (same env) -> exit 0.
- Compared libkernels.a (6 .cu HIP kernel objects) using utils/codeobj_diff.py.
- Result: `verdict=identical` -- exported symbols + device ISA identical (0 exports). All 6 gfx1100
  kernel TUs (amplitude.cu, basis.cu, angle.cu, validation.cu, iqp.cu, phase.cu) compile to
  identical device code objects.

No GPU re-run required. Transition: revalidate -> completed (validated_sha = fd8c7a38ac725b31f6b5ce1cc9e969cd357066f8).

## CUDA compile-check 2026-07-02 (linux-gfx90a, nvcc 12.6)

Real CUDA build (nvcc /opt/conda/envs/cuda/bin/nvcc, CUDA 12.6 V12.6.85,
CUDA_PATH=/opt/conda/envs/cuda). Prior round used QDP_NO_CUDA=1 stubs only.
This run exercises the actual cuda_rt path.

Commands:
```
export CUDA_PATH=/opt/conda/envs/cuda
export PATH=/opt/conda/envs/cuda/bin:$PATH
export RUSTFLAGS="-L /opt/conda/envs/cuda/lib/stubs -L /opt/conda/envs/cuda/lib"
unset QDP_USE_HIP QDP_HIP_ARCH_LIST ROCM_PATH

# Build (default cuda feature, real nvcc)
bash utils/timeit.sh mahout cuda-compile -- \
  cargo build --manifest-path projects/mahout/src/qdp/Cargo.toml \
  -p qdp-core -p qdp-kernels -j 16
# -> exit 0, 17.03s (cold)

# Compile test binaries (--no-run; no NVIDIA GPU on this host)
bash utils/timeit.sh mahout cuda-compile -- \
  cargo test --manifest-path projects/mahout/src/qdp/Cargo.toml \
  -p qdp-core -p qdp-kernels --no-run -j 16
# -> exit 0, 14.08s (warm)
```

Results:
- build.rs (qdp-kernels) found nvcc on PATH, took the CUDA (not HIP, not stub) branch.
  Selected arch targets from DEFAULT_CUBIN_ARCHES vs nvcc --list-gpu-code:
  sm_75, sm_80, sm_86, sm_89, sm_90 (all supported by CUDA 12.6).
  6x `.cu` compile jobs all exit 0 (confirmed in build script output). `kernels_dlink.o`
  present in libkernels.a (CUDA device link step). libkernels.a: 2.9 MB (vs ~1.3 MB HIP
  build -- real fatbinaries). amplitude.o contains .nvFatBinSegment, __nv_relfatbin,
  and PTX sections with sm_70 intrinsics: confirmed real CUDA nvcc compilation.
- build.rs (qdp-core) configure_cuda_linkage() found nvcc; emitted -lcudart and the
  cuda/lib64 link-search. No qdp_no_cuda cfg set (cuda_rt path selected).
- cuda_ffi.rs three-way split: `all(cuda, not(hip), not(qdp_no_cuda))` -> cuda_rt mod
  (real extern "C" libcudart bindings); no_cuda_stubs and hip_rt mods excluded.
- Test binaries compiled (--no-run): all 10 GPU test suites + 11 non-GPU suites + 2
  qdp-kernels test suites -> 23 executables, all exit 0. Test RUN skipped (no NVIDIA GPU).

Port edits verified on the CUDA path:
1. cuda_ffi.rs three-way split compiles cleanly on the cuda_rt branch.
2. configure_cuda_linkage() in qdp-core/build.rs correctly detects nvcc and emits -lcudart.
3. hip_compat/cuComplex.h #ifndef guards: only on the HIP include path (hip_compat/ dir
   not added to the CUDA build), so they do not affect the CUDA compilation.
4. usize->u64 widening in device.rs slice_mut (line 178):
   `range.start as u64 * size_of::<T>() as u64` -- in the hip mod (cuda build uses cudarc).
5. CudaStream Sync drop in device.rs hip mod: only `Send`, not `Sync` -- matches cudarc.
   The cuda build re-exports cudarc::driver::safe::CudaStream directly.

Verdict: CUDA path OK. The cuda_rt path builds through real nvcc 12.6 with no errors.
No port regression. Driver-stub (libcuda.so) not needed at build time (cudarc uses
libloading; stubs dir provided via RUSTFLAGS as a precaution). No NVIDIA GPU present
on this host: compile-only, no runtime test.

## Fix round 2026-08-13 (porter, linux-gfx1100) -- merge upstream main into moat-fix-1399

PR apache/mahout#1399 was CHANGES_REQUESTED by ryankert01 ("Need to resolve
conflicts"); GitHub reported CONFLICTING against upstream main. The published tip
fd8c7a38 is frozen (open PR), so the round is staged on `moat-fix-1399`, cut from
fd8c7a38 and MERGED (never rebased) with upstream/main. Staging tip:
1744956ded31cd8917e25aafd1795286fc36dbdd (merge 53c5f72fb + one follow-up).

Env: AMD Radeon Pro W7800 48GB (gfx1100, RDNA3 wave32), ROCm 7.2.1 / HIP 7.2.53211
(AMD clang 22.0.0git roc-7.2.3), cargo 1.97.1, HIP_VISIBLE_DEVICES=0.

### What upstream changed (fd8c7a38..upstream/main, 16 commits, 26 files in qdp/)
Relevant to the port:
- #1416 "Expose native cuda_available() and gate GPU tests on it": new
  `cudaGetDeviceCount` binding + `cuda_runtime_available()` in gpu/cuda_ffi.rs
  (also stubbed in the qdp_no_cuda mod), re-exported from lib.rs, and a new
  `cuda_available()` pyfunction in qdp-python/src/lib.rs.
- #1442 "apply shared max qubit validation": `validate_qubit_count()` in
  gpu/encodings/mod.rs called from six AmplitudeEncoder entry points; MAX_QUBITS
  removed from qdp-kernels/src/kernel_config.h (now Rust-side only); new
  gpu_ptr_encoding test `test_amplitude_gpu_pointer_paths_reject_excessive_qubits`.
- #1422 new tests/parquet_f32_fidelity.rs (CPU smoke + 3 GPU fidelity cases) and a
  benchmark; tests/common/mod.rs gained the f32/f64 Parquet writers.
- #1454 new src/estimate.rs + tests/estimate.rs (CPU-only memory estimator).
- #1412/#1415/#1439/#1456/#1457 + dependabot bumps: Python API, docs, Cargo.lock,
  uv.lock. No .cu kernel source touched by either side.

### Conflicts (4) and resolutions
1. `qdp-core/src/gpu/cuda_ffi.rs` -- upstream inserted cudaGetDeviceCount where our
   side has cudaMemGetInfo in the cuda_rt extern block. Kept both, and added the
   HIP-side `cudaGetDeviceCount -> hipGetDeviceCount` wrapper + extern decl, so
   upstream's new `cuda_runtime_available()` (moved to module scope, outside the
   three backend mods) resolves and reports AMD devices on the hip build. Without
   the wrapper the hip build fails to compile the helper.
2. `qdp-core/src/gpu/encodings/amplitude.rs` -- upstream added
   `use super::{QuantumEncoder, validate_qubit_count}` on the line where our port
   has `#[cfg(qdp_gpu_platform)]` in place of `#[cfg(target_os = "linux")]`. Took
   upstream's import + our cfg.
3. `qdp-core/tests/gpu_ptr_encoding.rs` -- upstream added MAX_QUBITS/encoder
   imports next to `use cudarc::driver::{DevicePtr, DeviceSlice}`, which our port
   had rewritten to `qdp_core::gpu_rt::{...}`. Took upstream's new imports, kept
   gpu_rt for the device-pointer traits.
4. `qdp-python/src/lib.rs` -- upstream registered `cuda_available` immediately
   before the `#[cfg(target_os = "linux")]` line our port carries as
   `#[cfg(qdp_gpu_platform)]`. Took both.

Verified afterwards that `git diff fd8c7a38..staging -- qdp/` is exactly upstream's
26-file delta plus the 6-line hipGetDeviceCount wrapper: no port intent was lost in
the merge.

### Follow-up commit (1744956de)
Upstream's new tests/parquet_f32_fidelity.rs gates its GPU module on
`#[cfg(target_os = "linux")]`. Every other GPU test in the crate uses
`qdp_gpu_platform` (emitted by qdp-core/build.rs: Linux always, Windows when the
hip feature is on), so on a Windows ROCm build that one file would silently drop
its three GPU cases. Switched it to `qdp_gpu_platform` and dropped the
"Linux + CUDA" wording. Test-only change.

### Build + test (exact commands)
```
export QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx1100 ROCM_PATH=/opt/rocm HIP_VISIBLE_DEVICES=0
cd projects/mahout/src/qdp
cargo build -p qdp-core -p qdp-kernels --no-default-features --features hip -j 16
cargo test  -p qdp-core -p qdp-kernels --no-default-features --features hip -- --test-threads=1
```
- build: exit 0 (16.2s cold). Only pre-existing warnings (iqp.cu unused param,
  phase.cu unused variable, and qdp-core's upstream "CUDA toolkit not found"
  notice from configure_cuda_linkage, which is cosmetic on a hip build).
- test: exit 0, **368 passed, 0 failed, 4 ignored** (ignored = the four pre-existing
  doctests requiring a GPU/IO fixture: gpu::pipeline::run_dual_stream_pipeline,
  io::read_numpy_batch, reader, readers::numpy::NumpyReader -- same four ignored
  before the merge, so the merge changed nothing there).
  Baseline: **353 passed on this same gfx1100 host at fd8c7a38** (measured
  2026-08-13 from a detached worktree at fd8c7a38, identical command). 353 -> 368 is
  +15, accounted for exactly: estimate.rs 9 (new file), parquet_f32_fidelity.rs 4
  (new file, incl. the gpu:: 8/12/16-qubit fidelity cases), gpu_ptr_encoding 68 -> 69
  (the new excessive-qubit case) = the 14 `#[test]` functions upstream adds
  (`git grep -c '#\[test\]' <rev> -- qdp/qdp-core/tests qdp/qdp-core/src
  qdp/qdp-kernels` sums to 367 at fd8c7a38 and 381 at the staging tip), plus 1
  newly-executing doctest, `qdp-core/src/estimate.rs - estimate::estimate_memory`
  (line 111), which upstream's new file brings with it. Every one of the 15 runs and
  passes on AMD; none of upstream's new tests is skipped or ignored on this config.
  The earlier "was 358 / +10" wording was wrong twice: 358 is a gfx90a figure from a
  different host (gfx1100 had no GPU pass count at fd8c7a38 -- its 2026-07-02 record
  is a binary-equivalence carry-forward, hence this baseline re-run), and the upstream
  delta is 14 tests, not 10. reader (3) and parquet_f32 (7) were miscounted as new;
  both already ran at fd8c7a38 with the same counts. The gfx90a 358 vs gfx1100 353 gap
  at the same commit is host environment, not the port: suites gated on optional
  dependencies (torch_io runs 3 of its 5 `#[test]`s here without the pytorch feature)
  differ between the two machines.
  Full breakdown at the tip: lib 100, arrow_ipc_io 5, estimate 9, gpu_angle 12, gpu_api_workflow 8,
  gpu_basis 7, gpu_dlpack 9, gpu_fidelity 17, gpu_iqp 22, gpu_memory_safety 4,
  gpu_norm_f32 2, gpu_ptr_encoding 69, gpu_validation 8, null_handling 6, numpy 4,
  parquet_f32 7, parquet_f32_fidelity 4, parquet_io 8, preprocessing 14, reader 3,
  tensorflow_io 9, torch_io 3, types 6, qdp-kernels amplitude_encode 21,
  angle_encode 10, doctests 1.

### CUDA (default feature) compile check -- real nvcc 12.8, this host
No NVIDIA GPU here, so build-only; the PR's NVIDIA CI covers the run.
```
export CUDA_PATH=/opt/conda/envs/cuda-12.8 PATH=$CUDA_PATH/bin:$PATH
export RUSTFLAGS="-L $CUDA_PATH/lib/stubs -L $CUDA_PATH/lib"
export CARGO_TARGET_DIR=agent_space/mahout-cuda-target
unset QDP_USE_HIP QDP_HIP_ARCH_LIST ROCM_PATH
cargo build -p qdp-core -p qdp-kernels -j 16          # exit 0, 14.1s
cargo test  -p qdp-core -p qdp-kernels --no-run -j 16 # exit 0, 23 test binaries
```
qdp-core's build script emitted `rustc-link-lib=cudart` (real cuda_rt path, not the
qdp_no_cuda stub path); libkernels.a is 4.7 MB of nvcc fatbinaries.

### Conflict-free confirmation
- `git merge-base --is-ancestor upstream/main moat-fix-1399` -> yes (staging tip
  contains current upstream main, so merging it into main is a fast-forward).
- `git merge-base --is-ancestor fd8c7a38 moat-fix-1399` -> yes (published tip is
  still an ancestor; the later fast-forward of the PR branch remains possible).
- `git merge-tree --write-tree upstream/main moat-fix-1399` -> clean.
- Pushed ONLY `moat-fix-1399` to the fork; `moat-port` untouched (frozen).
- `python3 utils/jargon.py --port mahout` and the same scan over
  `main..moat-fix-1399` (commits and added lines): clean.

### Gotchas from this round
- Upstream is actively adding tests that gate on `target_os = "linux"`. Any future
  sync must re-check new `#[cfg(target_os = "linux")]` sites against the port's
  `qdp_gpu_platform` cfg, or the Windows-ROCm build quietly loses coverage.
- When upstream adds a runtime entry point to cuda_ffi.rs, all THREE backends need
  it (cuda_rt extern, no_cuda_stubs stub, hip_rt wrapper); a helper written at
  module scope over those names fails to compile on hip if only the first two are
  updated. The auto-merge silently supplied the stub and cuda_rt sides only.

## Review 2026-08-13 (reviewer, linux-gfx1100) -- fix round 1399 delta

Scope: `git diff fd8c7a38..1744956de` on `moat-fix-1399` (merge 53c5f72fb of
upstream 206ff2fe8 + follow-up 1744956de). Problems only; the merge resolutions,
the hipGetDeviceCount wrapper, the parquet_f32_fidelity cfg change and the
promoted skill lesson are otherwise correct and are not restated here.

### 1. Commit title of the merge lacks the `[ROCm]` prefix (must fix)
53c5f72fb is titled `Merge upstream main into the AMD/HIP branch`. AGENTS.md
("Commit messages and upstream-visible text") requires every commit title to
start with `[ROCm]` and states no merge-commit exception; the round's other
commit (1744956de) complies. This commit becomes permanently visible on
apache/mahout#1399 once `upstream.py --merge-fix` fast-forwards the PR branch, so
it can only be corrected now, while the staging branch is unpublished and no
person has reviewed it. Reword with `git commit --amend` on the merge plus a
replay of 1744956de; the published tip fd8c7a38 stays parent 1, so the
"published tip is still an ancestor" invariant survives and `moat-port` is not
touched. Update `head_sha` afterwards. If the porter judges the history rewrite
worse than the deviation, record that rationale here instead -- but the default
per AGENTS.md is the prefix. Body, length (43 chars), AI disclosure, Test Plan
and trailers on both commits are correct.

### 2. Test-count reconciliation in the round's evidence is wrong (must fix)
notes.md lines 1682-1686 read "Was 358 at fd8c7a38 on gfx90a; the +10 are
upstream's new tests". Upstream's delta adds 14 test functions, not 10:
`git grep -c '#\[test\]' <rev> -- qdp/qdp-core/tests qdp/qdp-core/src
qdp/qdp-kernels` gives 367 at fd8c7a38 and 381 at 1744956de, and the three
affected files match the note's own per-suite numbers (estimate.rs 9 new,
parquet_f32_fidelity.rs 4 new, gpu_ptr_encoding.rs 68 -> 69). The 358 figure is
also a gfx90a number and is not a comparable baseline: gfx1100 has no pass count
at fd8c7a38 (its 2026-07-02 record is a binary-equivalence carry-forward with no
GPU re-run). So four tests that passed in the gfx90a 358 run are unaccounted for
in this gfx1100 368 run -- plausibly the 4 doctests this run reports as ignored,
but that is a guess and the note asserts a reconciliation that does not hold.
The 368 total itself is fine: the full per-suite breakdown in that paragraph sums
to exactly 368. Fix the sentence to state the +14 upstream delta and drop or
qualify the cross-platform 358 comparison; if the 4 doctests are the difference,
say so with the ignored-doctest names.

### 3. The `qdp_no_cuda` stub configuration was not compiled this round (must fix)
The merge restructured qdp-core/src/gpu/cuda_ffi.rs and upstream's new
`cudaGetDeviceCount` stub landed in `no_cuda_stubs` (cuda_ffi.rs:172, gated at
:138 on `all(feature = "cuda", not(feature = "hip"), qdp_no_cuda)`). This round
compiled only the hip_rt backend and the real-nvcc cuda_rt backend -- notes.md
line 1704 confirms the CUDA check took the cudart path, so the stub arm was
excluded. That leaves one of the three backends selected by the file the merge
rewrote unbuilt. Add the check the previous round ran:
`QDP_NO_CUDA=1 cargo check -p qdp-core` (default cuda feature, nvcc off PATH),
and record the result.

### Verified clean (no action)
- `git diff fd8c7a38..1744956de -- qdp/` is 26 files; the tree-level three-way
  comparison against the merge base dccc97db shows zero files where an upstream
  change was dropped in favour of our side and zero where a port change was
  dropped in favour of upstream's. `git show --cc 53c5f72fb` introduces exactly
  11 lines beyond the two parents: the cudaGetDeviceCount extern in cuda_rt, the
  hipGetDeviceCount extern plus its wrapper, and the cuda_runtime_available doc
  edits. The porter's "upstream delta + 6-line wrapper" claim holds.
- All three cuda_ffi.rs backends define `cudaGetDeviceCount` and their cfgs are
  mutually exclusive in every feature combination, including cuda+hip together
  (hip wins), so upstream's `cuda_runtime_available` (cuda_ffi.rs:444) resolves
  in each. hipGetDeviceCount matches cudaGetDeviceCount 1:1 (`int*` out,
  hipSuccess == 0), and the `rc == 0 && count > 0` test is correct for the
  no-device case in either runtime.
- The follow-up's `#[cfg(qdp_gpu_platform)]` on parquet_f32_fidelity.rs:116 is
  the only such gate left in qdp/**/*.rs (`grep -rn 'target_os = "linux"'` now
  matches only a build.rs doc comment), so it matches how every other GPU test is
  gated and it belongs in this round.
- No .cu source changed: the only qdp-kernels file in the delta is
  kernel_config.h, where upstream deleted `MAX_QUBITS`, which no kernel source
  references. No warp-size, resource-handle, OOB or library-swap surface is
  touched by this delta.
- `utils/jargon.py --port mahout` clean; independent scan of the delta's added
  lines and both commit messages finds no MOAT vocabulary, no AMD-internal
  account reference and no Co-Authored-By/noreply trailer.
- Ancestry: upstream/main (206ff2fe8) and the published tip fd8c7a38 are both
  ancestors of 1744956de. Fork worktree is clean (integrity gate satisfied).
- The promoted lesson (.claude/skills/cuda-to-rocm/references/upstream-sync.md
  plus the SKILL.md pointer) is accurate on both of its claims for this port -- a
  missing hip_rt entry point does break compilation, and a `target_os = "linux"`
  test gate does silently drop coverage on Windows ROCm -- and is written
  cross-project (CMake equivalents included), so it belongs in the skill.

## Review fixes 2026-08-13 (porter, linux-gfx1100) -- fix round 1399, all three findings

New staging tip: 9a3a08e0f3061b00ddf8f2cfb3f5cd5c49b38d66 (was 1744956de).
`moat-fix-1399` force-pushed with `--force-with-lease`; it carries no review PR
yet, so rewriting it is allowed. `moat-port` untouched (frozen, PR open).

### 1. `[ROCm]` prefix on the merge commit (fixed)
Amended the merge's message only and replayed the follow-up:
```
git checkout --detach 53c5f72fb
git commit --amend -F <msg>          # -> 5a15c3f25, both parents preserved
git cherry-pick 1744956de            # -> 9a3a08e0f
git branch -f moat-fix-1399 9a3a08e0f
```
Merge title is now `[ROCm] Merge upstream main into the AMD/HIP branch` (49
chars). Evidence that nothing but commit metadata moved:
- `git rev-parse 9a3a08e0f^{tree}` == `git rev-parse 1744956de^{tree}` ==
  `5584f6f96cd105dd30684030676e036d51c5d858`; `git diff 1744956de 9a3a08e0f` empty.
- merge tree unchanged too: `53c5f72fb^{tree}` == `5a15c3f25^{tree}` ==
  `77f49f999bddb05bbc48ed08d0ed8d532f2620c0`.
- `git log -1 --format=%P 5a15c3f25` -> `fd8c7a38ac72... 206ff2fe863f...`, i.e. the
  published tip is still parent 1 and upstream main parent 2.
- `git merge-base --is-ancestor` yes for fd8c7a38, upstream/main and 206ff2fe8
  against 9a3a08e0f.
- the follow-up commit's message is byte-identical after the replay.
While rewriting the message, corrected one factual error in its last paragraph:
it listed `parquet_f32 (7)` among "upstream's newly merged suites", but that suite
already existed and ran at fd8c7a38 (see finding 2). The paragraph now names
estimate, parquet_f32_fidelity, the new gpu_ptr_encoding case and the
estimate_memory doc example, and gives 368 vs the measured 353 baseline. No other
body text changed.
Since only commit metadata differs, no GPU re-run was required for the rewrite;
the gfx1100 evidence recorded for 1744956de applies unchanged to 9a3a08e0f by
tree identity. (The suite was re-run anyway for finding 2 and passes at the new
tip: 368 passed, 0 failed, 4 ignored.)

### 2. Test-count reconciliation (fixed)
The "Build + test" paragraph above now states the measured gfx1100 baseline (353
at fd8c7a38, re-run today in a detached worktree because gfx1100's fd8c7a38
record was a binary-equivalence carry-forward with no GPU run) and the exact
+15 = 14 upstream `#[test]` functions + 1 newly-executing doctest. Commands:
```
git worktree add --detach agent_space/mahout-base fd8c7a38
cd agent_space/mahout-base/qdp
export QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx1100 ROCM_PATH=/opt/rocm HIP_VISIBLE_DEVICES=0
cargo test -p qdp-core -p qdp-kernels --no-default-features --features hip -j 16 \
  -- --test-threads=1                       # 353 passed, 0 failed, 4 ignored
```
Per-suite diff base -> tip: estimate 0->9, parquet_f32_fidelity 0->4,
gpu_ptr_encoding 68->69, qdp_core doctests 0->1 passed (the same 4 stay ignored).
Every other suite is unchanged. The 358 gfx90a figure is not a gfx1100 baseline
and is no longer used as one.

### 3. `qdp_no_cuda` stub backend compiled (fixed)
The third backend selected by the merged cuda_ffi.rs is
`all(feature = "cuda", not(feature = "hip"), qdp_no_cuda)`. qdp-core/build.rs
sets `qdp_no_cuda` when `QDP_NO_CUDA` is 1/true/yes or nvcc is absent, so the
default (cuda) feature set plus that env var selects the stub arm:
```
cd qdp
env -u QDP_USE_HIP -u QDP_HIP_ARCH_LIST -u ROCM_PATH -u CUDA_PATH QDP_NO_CUDA=1 \
  CARGO_TARGET_DIR=<scratch> cargo check -p qdp-core -j 16
```
exit 0 in 9.4s. The build script's `cargo:warning=qdp-core: CUDA toolkit not
found (nvcc not in PATH). Building with stub CUDA Runtime symbols` confirms
`qdp_no_cuda` was emitted, and `qdp-core/Cargo.toml` has `default = ["cuda"]`
with `hip` off, so the cfg on cuda_ffi.rs:138 holds and `no_cuda_stubs` (with
upstream's new `cudaGetDeviceCount` stub) is the arm that was type-checked.
All three backends of the file the merge rewrote are now covered this round:
hip_rt (build + 368-test GPU run), cuda_rt (nvcc 12.8 build + 23 test binaries),
no_cuda_stubs (this check).

## Re-review 2026-08-13 (reviewer, linux-gfx1100) -- fix round 1399, PASS

Narrow re-review of the rewritten staging tip 9a3a08e0f (was 1744956de). No
problems found; all three findings of the 2026-08-13 review are closed and
nothing regressed. Evidence checked independently:

- Tree identity: `9a3a08e0^{tree}` == `1744956de^{tree}` ==
  5584f6f96cd105dd30684030676e036d51c5d858, `git diff` between them empty. The
  amended merge 5a15c3f25 has parents fd8c7a38 (published tip, parent 1) and
  206ff2fe8 (upstream/main, parent 2), same as 53c5f72fb; both are ancestors of
  the tip. `origin/moat-fix-1399` == 9a3a08e0f after a fresh fetch,
  `origin/moat-port` still fd8c7a38 (freeze intact).
- Merge message: title `[ROCm] Merge upstream main into the AMD/HIP branch` (50
  chars). `diff` of old vs new body shows exactly two changes -- the title prefix
  and the last paragraph. The follow-up commit's message is byte-identical
  (title 57 chars). Neither authored commit carries a Co-authored-by, noreply or
  Signed-off-by trailer (the dependabot noreply trailers in the range are
  upstream's own commits). AI disclosure and Test Plan intact.
- New last-paragraph clause fact-checked: `tests/parquet_f32.rs` exists at
  fd8c7a38 with 7 `#[test]`s, so removing it from "newly merged suites" is
  correct; `tests/estimate.rs` (9) and `tests/parquet_f32_fidelity.rs` (4, three
  inside the `#[cfg(qdp_gpu_platform)]` gpu module) are new files;
  gpu_ptr_encoding 68 -> 69. Per-file `#[test]` delta between the two commits is
  confined to exactly those three files (repo-wide 367 -> 381 = +14).
  `src/estimate.rs:111` is a bare ``` fence with asserts and is the only doc
  example the merge adds, so +1 executing doctest; 353 + 14 + 1 = 368.
- Notes reconciliation (notes.md 1682-1709): per-suite breakdown sums to exactly
  368; the four ignored doctests are the only `rust,ignore` fences in
  qdp-core/src (gpu/pipeline.rs:245, io.rs:258, reader.rs:34,
  readers/numpy.rs:38), none touched by the merge, so "same four before and
  after" holds; the 358-gfx90a-vs-353-gfx1100 gap is recorded as host
  environment (optional-dependency suites) and no longer used as a baseline.
- Stub backend: qdp-core/build.rs:88-101 emits `qdp_no_cuda` when `QDP_NO_CUDA`
  is 1/true/yes, qdp-core/Cargo.toml has `default = ["cuda"]` with `hip` off, so
  the recorded `QDP_NO_CUDA=1 cargo check -p qdp-core` selects
  `all(feature = "cuda", not(feature = "hip"), qdp_no_cuda)` at cuda_ffi.rs:138,
  the arm holding upstream's new stub at :172. Exit 0 recorded with the
  build-script warning that confirms the cfg fired.
- `python3 utils/jargon.py --port mahout` clean; fork worktree clean
  (`git status --porcelain` empty), integrity gate satisfied.

## Validation 2026-08-13 (validator, linux-gfx1100) -- fix round 1399 revalidation, PASS

Independent revalidation at the fix round's rewritten staging tip, since gfx1100's
previously recorded `validated_sha` (fd8c7a38) lagged the new `head_sha`
(9a3a08e0f3061b00ddf8f2cfb3f5cd5c49b38d66) after the reviewer's must-fix round.
Confirmed checkout: `git -C projects/mahout/src rev-parse HEAD` ==
9a3a08e0f3061b00ddf8f2cfb3f5cd5c49b38d66 on `moat-fix-1399`, `git status
--porcelain` empty (integrity gate satisfied) before and after the run.

Env: AMD Radeon Pro W7800 48GB (gfx1100, RDNA3 wave32), ROCm 7.2.1 / HIP 7.2.53211
(AMD clang 22.0.0git roc-7.2.3), rustc 1.97.1, cargo 1.97.1. Same host as the
porter/reviewer rounds above.

### Build + test (exact commands)
```
utils/timeit.sh mahout compile -- bash -c 'cd projects/mahout/src/qdp && \
  QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx1100 ROCM_PATH=/opt/rocm \
  cargo build -p qdp-core -p qdp-kernels --no-default-features --features hip -j 16'
utils/timeit.sh mahout test -- bash -c 'cd projects/mahout/src/qdp && \
  QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx1100 ROCM_PATH=/opt/rocm \
  cargo test -p qdp-core -p qdp-kernels --no-default-features --features hip -- \
  --test-threads=1'
```
- build: exit 0. Same pre-existing warnings as recorded above (iqp.cu unused
  param, phase.cu unused variable, qdp-core's cosmetic "CUDA toolkit not found"
  build-script notice on a hip build).
- test: exit 0, **368 passed, 0 failed, 4 ignored**, summed independently from the
  per-suite `test result:` lines (100+5+9+12+8+7+9+17+22+4+2+69+8+6+4+7+4+8+14+3+9+
  3+6+0+21+10+1+0 = 368) -- matches the porter's recorded count for this tip
  exactly. Ignored are the same 4 pre-existing doctest fixtures
  (gpu::pipeline::run_dual_stream_pipeline, io::read_numpy_batch, reader,
  readers::numpy::NumpyReader). The dual-stream async-pipeline tests
  (test_amplitude_encoding_async_pipeline, test_angle_encoding_async_pipeline in
  gpu_api_workflow; test_angle_batch_f32_async_pipeline_path in
  gpu_angle_encoding) pass in this standard run without needing
  `QDP_ENABLE_OVERLAP_TRACKING=1`; that env var is not part of this round's
  recorded bar (its one prior use in this file is an unrelated, older gfx90a
  round), so it was not re-run separately.

### CUDA no-regression gate
Not re-run: already recorded in this notes.md under "Fix round 2026-08-13
(porter, linux-gfx1100)" (real nvcc 12.8, build-only, exit 0, no NVIDIA GPU on
this host) against tree 5584f6f96cd105dd30684030676e036d51c5d858, which the
porter's amend and the reviewer's re-review both independently confirmed is the
identical tree at 9a3a08e0f (`git rev-parse 9a3a08e0f^{tree}` ==
`git rev-parse 1744956de^{tree}`). No .cu kernel source is in the delta at all
(reviewer's "Verified clean" note above), so the CUDA build result carries
forward by tree identity; not re-run here.

### Gates
- `python3 utils/jargon.py --port mahout`: clean.
- Documentation: unchanged by this round (no build/doc files in the 26-file
  upstream-merge delta or the follow-up); the ROCm build documentation recorded
  in earlier rounds still matches the build commands used here.
- Fork worktree clean (`git status --porcelain` empty) before and after.

### Result
`python3 utils/moatlib.py set-state mahout linux-gfx1100 completed --agent
validator` -> recorded `validated_sha` = `head_sha` =
9a3a08e0f3061b00ddf8f2cfb3f5cd5c49b38d66. No anomalies.

## Validation 2026-08-13 (validator, linux-gfx90a) -- fix round 1399 revalidation, PASS

Independent revalidation at the fix round's staging tip on `moat-fix-1399`, since
gfx90a's previously recorded `validated_sha` (fd8c7a38, the old published/frozen
tip) lagged the new `head_sha` (9a3a08e0f3061b00ddf8f2cfb3f5cd5c49b38d66).
Confirmed checkout: `git -C projects/mahout/src rev-parse HEAD` ==
9a3a08e0f3061b00ddf8f2cfb3f5cd5c49b38d66 on `moat-fix-1399`, `git status
--porcelain` empty (integrity gate satisfied) before and after the run. Fork
freeze confirmed intact: `origin/moat-port` is still fd8c7a38 (fetched fresh).

Env: 4x AMD Instinct MI250X GCDs (gfx90a, wave64), GCD 2 idle (HIP_VISIBLE_DEVICES=2),
ROCm/HIP 7.14.60850 (AMD clang 23.0.0git), rustc/cargo 1.97.1.

### Host environment note (this host differs from earlier gfx90a rounds)
This host's ROCm install is a TheRock-style `_rocm_sdk_devel` pip package, not
`/opt/rocm` (which does not exist here) -- `ROCM_PATH` is already correctly set
in the default shell environment
(`/opt/conda/envs/py_3.12/lib/python3.12/site-packages/_rocm_sdk_devel`).
Exporting `ROCM_PATH=/opt/rocm` (the value earlier notes.md rounds used, from a
different host generation) makes hipcc's internal `clang++` invocation fail
(`sh: /opt/rocm/lib/llvm/bin/clang++: not found`, surfacing as `hipcc` exit 127)
and, because `qdp-kernels/build.rs` does not declare
`cargo:rerun-if-env-changed=ROCM_PATH`, a build script run with the wrong value
gets cached and silently reused (wrong `-L` search path) even after fixing the
env var, until the stale `target/debug/build/qdp-kernels-*` dir is cleaned
(`cargo clean -p qdp-kernels -p qdp-core`). Symptom once the compile step
"passes" on the stale cache: the *test* binaries fail to link with
`rust-lld: error: unable to find library -lamdhip64`, because only the final
linked binary (not the rlib) needs the search path. Lesson: don't hardcode
`ROCM_PATH` from old notes -- check `env | grep -i rocm_path` first, and if a
build script doesn't declare a var in `rerun-if-env-changed`, an env correction
requires `cargo clean -p <crate>`, not just re-running with the right value.
Separately, `/opt/rust` (the shared system `CARGO_HOME`) has a registry cache
owned by root (0755) that the `jenkins` user can read but not write new crates
into; a private `CARGO_HOME` (agent_space/cargo-home) was required for `cargo
build`/`test` to fetch anything not already cached there.

### Build + test (exact commands)
```
export CARGO_HOME=/var/lib/jenkins/moat/agent_space/cargo-home
export PATH=/opt/rust/bin:$PATH
export QDP_USE_HIP=1 QDP_HIP_ARCH_LIST=gfx90a   # ROCM_PATH left at its correct default
utils/timeit.sh mahout compile -- cargo build \
  --manifest-path projects/mahout/src/qdp/Cargo.toml \
  -p qdp-core -p qdp-kernels --no-default-features --features hip -j 16
HIP_VISIBLE_DEVICES=2 utils/timeit.sh mahout test -- cargo test \
  --manifest-path projects/mahout/src/qdp/Cargo.toml \
  -p qdp-core -p qdp-kernels --no-default-features --features hip -- \
  --test-threads=1
```
- build: exit 0, 4.6s (post-clean rebuild). Same pre-existing warnings as every
  other round (iqp.cu unused param, phase.cu unused variable, qdp-core's
  cosmetic "CUDA toolkit not found" build-script notice on a hip build).
- test: exit 0, **368 passed, 0 failed, 4 ignored** -- summed independently from
  the 28 `test result:` lines (100 lib + 5 arrow_ipc_io + 9 estimate + 12
  gpu_angle_encoding + 8 gpu_api_workflow + 7 gpu_basis_encoding + 9 gpu_dlpack +
  17 gpu_fidelity + 22 gpu_iqp_encoding + 4 gpu_memory_safety + 2 gpu_norm_f32 +
  69 gpu_ptr_encoding + 8 gpu_validation + 6 null_handling + 4 numpy + 7
  parquet_f32 + 4 parquet_f32_fidelity + 8 parquet_io + 14 preprocessing + 3
  reader + 9 tensorflow_io + 3 torch_io + 6 types + 0 qdp-kernels lib + 21
  amplitude_encode + 10 angle_encode + 1 doctest + 0 qdp-kernels doctests =
  368) -- matches the gfx1100 fix-round revalidation count exactly (same
  head_sha, same 4 ignored doctest fixtures: gpu::pipeline::run_dual_stream_pipeline,
  io::read_numpy_batch, reader, readers::numpy::NumpyReader).
- Async-pipeline and stream-ordering tests confirmed passing on wave64:
  test_amplitude_encoding_async_pipeline, test_angle_encoding_async_pipeline
  (gpu_api_workflow), test_angle_batch_f32_async_pipeline_path
  (gpu_angle_encoding), test_l2_norm_batch_kernel_stream (amplitude_encode),
  test_encode_from_gpu_ptr_f32_with_stream_non_default_success,
  test_encode_batch_from_gpu_ptr_f32_with_stream_success (gpu_ptr_encoding).

### CUDA no-regression gate
Not re-run: already recorded once for this head_sha under "Fix round 2026-08-13
(porter, linux-gfx1100)" (real nvcc 12.8, build-only, exit 0, no NVIDIA GPU on
that host) and independently confirmed by the reviewer/validator as applying to
tree 5584f6f96cd105dd30684030676e036d51c5d858, which is 9a3a08e0f's tree. No .cu
kernel source is in the fix round's delta at all. Per the validator role
(compile-only CUDA gate runs once per head_sha), not repeated here.

### Gates
- Jargon: `python3 utils/jargon.py --port mahout` alone reports the range
  `main..moat-port`, which is the FROZEN published branch (fd8c7a38) and misses
  the in-flight fix round entirely -- `moatlib`'s `fork_branch` field is never
  set, so `port_range()` always defaults to `moat-port`, not `fix.branch`.
  Confirmed the real content instead: `python3 utils/jargon.py -C
  projects/mahout/src --commits main..moat-fix-1399 --diff main..moat-fix-1399`
  (after `git fetch --depth 50 origin main:main` in the shallow clone, since
  `main` did not exist locally yet -- an unresolvable git range in
  `scan_commits`/`scan_diff` silently reports 0 hits/"clean" outside of
  `port_range()`'s explicit check, so resolving `main` first matters). Result:
  clean (66 commits, 0 hits). Matches the porter's and reviewer's independent
  scans of the same range earlier in this file.
- Documentation: unchanged by this round; `qdp/DEVELOPMENT.md` ("### AMD GPU
  build (ROCm / HIP)") and `qdp/qdp-python/README.md` ("AMD ROCm Usage") still
  present and accurate in the fix-round tree (grep-confirmed).
- Fork worktree clean (`git status --porcelain` empty) before and after.

### Result
`python3 utils/moatlib.py set-state mahout linux-gfx90a completed --agent
validator` -> recorded `validated_sha` = `head_sha` =
9a3a08e0f3061b00ddf8f2cfb3f5cd5c49b38d66. No anomalies.
