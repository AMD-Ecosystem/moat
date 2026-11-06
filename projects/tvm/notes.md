# tvm notes

## Intake screen (2026-11-05, windows-gfx1151 host, gap-matrix seed)

Upstream: https://github.com/apache/tvm (Apache-2.0, tier 1, 13.9k stars, not archived, pushed 2026-11-03). Screened at a shallow clone of main on 2026-11-05.

### Licence

- Top level: Apache-2.0 (https://github.com/apache/tvm/blob/main/LICENSE). Recorded with `set-license-spdx`.
- `python3 utils/licenses.py scan-nvidia agent_space/tvm-screen`: no NVIDIA proprietary licence text in the main tree.
- Submodules (not cloned by the shallow screen, checked by GitHub licence field): `3rdparty/cutlass` (NVIDIA/cutlass, field NOASSERTION; the repo's LICENSE.txt is BSD-3-Clause), `3rdparty/cutlass_fpA_intB_gemm` (tlc-pack, Apache-2.0), `3rdparty/libflash_attn` (tlc-pack, BSD-3-Clause), `3rdparty/tvm-tirx` (apache/tvm-tirx, Apache-2.0). All are CUDA-only optional contribs except tvm-tirx; none would be modified by the scope below. If the planner touches cutlass, run scan-nvidia on the submodule checkout first.

### What the .cu source is

About 214 KB of `.cu`/`.cuh`, all under `src/runtime/extra/contrib/`, all optional (off by default):

- `cutlass/` (~120 KB): FP8/FP16 grouped and blockwise-scaled GEMMs for sm90/sm100 (Hopper/Blackwell only). No HIP equivalent for CUTLASS/CuTe; the AMD path in TVM for library GEMM is the hipBLAS/hipBLASLt BYOC (`TVM_USE_HIPBLAS`).
- `vllm/` (~59 KB): vLLM paged-attention and cache kernels, `TVM_USE_VLLM`, CUDA-only build. TVM's own TIR paged KV cache (`relax/frontend/nn/llm`) is the default and compiles for ROCm through codegen.
- `nvshmem/` (~24 KB): distributed GEMM and KV transfer (disaggregated serving), NVSHMEM-only.
- `thrust/thrust.cu` (23 KB): already builds for ROCm via rocThrust (`TVM_TIRX_USE_ROCM AND TVM_USE_THRUST`, cmake/modules/contrib/ROCM.cmake).
- `curand/` (1.4 KB): CUDA-only.
- FlashInfer integration (`python/tvm/relax/backend/cuda/flashinfer.py`) is CUDA-only.

The bulk of TVM's GPU work is generated code, not hand-written `.cu`.

### Existing AMD support (upstream, first-class)

- ROCm is a native TVM target: LLVM AMDGPU codegen + HIP runtime (`TVM_TIRX_USE_ROCM`, `libtvm_runtime_rocm.so`), now living in the `apache/tvm-tirx` submodule after the 2026 TIRx bring-up.
- ROCm Relax pipeline (`python/tvm/relax/backend/rocm/pipeline.py`), hipBLAS/hipBLASLt BYOC (`relax/backend/rocm/hipblas.py`, `src/runtime/extra/contrib/hipblas`), rocThrust sort/scan, RCCL (`TVM_USE_RCCL`), dlight GPU schedules with ROCm cases (`s_tir/dlight/gpu/{matmul,gemv,low_batch_gemv}.py`), `tensor_intrin/rocm.py`.
- Docs: docs/install/from_source.rst "Building ROCm support": "Currently, ROCm is supported only on linux".
- CI: `docker/Dockerfile.ci_gpu` installs ROCm 6.4.4 as a "cross compilation toolchain"; `tests/scripts/task_build_gpu_other.sh` builds `tvm_runtime_rocm`; `task_python_unittest_gpuonly.sh` lists `rocm` in `TVM_TEST_TARGETS`, but the runners are NVIDIA, so ROCm tests skip ("No device exists"). **No AMD GPU test coverage in upstream CI.**
- Maintenance: ROCm-relevant fixes merge regularly (#19390 ROCm-bundled LLVM 20 `insertDeclare`, 2026-04; #18706 same class, maintainer MasterJH5574 responded within a day on #18709). Open #20545 (oraluben, 2026-10-04) "[FIX][ROCm] Use HIP for device existence checks": replaces direct HSA use with `hipGetDeviceCount`, motivated by the Windows ROCm 10.0 pip SDK. That is a third party already working on ROCm-on-Windows for TVM.
- #19506 (hang in `ROCMDeviceAPI::GPUCopy` H2D on null stream, Ryzen AI HX 370 / gfx1150, ROCm 7.2.3): closed by the reporter as "a ROCm 7.2.3 issue", not root-caused in TVM.
- Downstream: MLC-LLM (built on TVM) ships ROCm wheels (`mlc-ai-nightly-rocm6x` from mlc.ai/wheels).

### Duplicate effort

- No ROCm-org or AMD-Ecosystem fork of tvm. GitHub forks owned by AMD-named personal accounts (glen-amd/tvm 2023-10, weiwangamd/tvm 2023-06, kkHuang-amd/tvm 2022-06) are stale. No hip/rocm/amd branches upstream.
- The `amd` and `AMD-AGI` orgs are behind SAML SSO for this token, so a fork or internal effort there was not checkable; that part of the duplicate check stays open.
- Upstream PR #20545 (Windows ROCm device check) is the only open AMD PR; any Windows work must build on it, not compete.

### Assessment

Mature, maintained upstream ROCm backend: this is not a CUDA-to-HIP port. What remains are (a) the validation gap -- nobody runs TVM's GPU tests on AMD hardware upstream, on current ROCm 7.x, or on wave32 RDNA; (b) Windows ROCm, already being started by #20545; (c) CUDA-only optional contribs (CUTLASS sm90/sm100, vLLM kernels, NVSHMEM, FlashInfer, cuRAND) that either have no HIP target or are superseded by TVM's codegen path and the hipBLAS BYOC -- not worth porting.

Upstream churn is very high (repeated large refactors by tqchen: runtime/compiler split, TIRx bring-up, backend DSO reorg), so a long-lived fork delta is costly; anything MOAT finds should be small fixes sent promptly.

### Recommendation

**fork, with a validate-and-fix scope only** (no source port): build `TVM_TIRX_USE_ROCM=ON` (+ hipBLAS, rocThrust) against current ROCm 7.x on wave64 and wave32 Linux hosts, run the GPU-target Python suites (`TVM_TEST_TARGETS=rocm`, `tests/python/{all-platform-minimal-test,codegen,relax,s_tir/dlight}`) and an MLC-LLM-style LLM compile smoke test, fix real failures in small upstream PRs, and add a docs line for the validated ROCm versions/arches. Windows: only after #20545 lands, contributing to that author's line. Explicitly out of scope: CUTLASS, vLLM, NVSHMEM, FlashInfer, cuRAND contribs.

If a person judges a validation-only scope not worth fleet time on a backend upstream already owns, the decline is `already-supported` (upstream has a native, maintained ROCm path; the only gap is AMD CI coverage).

Build is heavy (LLVM + large C++ tree; full builds go to a fleet host, not this laptop). No MOAT dependencies (`depends_on` empty); needs LLVM with the AMDGPU target and ROCm hipBLAS/hipBLASLt/rocThrust/rocPRIM.

### Question for jeff

Is a validation-only round (run TVM's GPU tests on wave64 + wave32 on current ROCm, fix what fails, no source port) on a mature upstream ROCm backend worth fleet time, or should this be declined as `already-supported`? The screen leans fork because upstream has zero AMD GPU test coverage, but there is no concrete known breakage at current main.
