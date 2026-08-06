# PhoenixOS Porting Plan

## Project

- **Name**: PhoenixOS
- **Upstream**: https://github.com/SJTU-IPADS/PhoenixOS
- **Default branch**: v0
- **Description**: OS-level GPU checkpoint/restore (C/R) system that transparently checkpoints processes using GPUs. First system to support concurrent C/R without stopping application execution.

## Existing AMD Support

**Status**: No existing HIP/ROCm port.

The upstream explicitly states ROCm support is "Developing" (README.md line 3), with only CUDA currently implemented. The ROCm documentation section (`docs/docs/source/rocm_gsg/index.rst`) is empty -- contains only a title with no content. The meson.build (line 57-61) enforces `conf_runtime_target != 'cuda'` -> assert failure, confirming ROCm is not implemented.

**Related work**: CRIU has a separate AMD GPU plugin (`plugins/amdgpu/`) for ROCm checkpoint/restore, but it is a completely different architecture (kernel driver-level via KFD) rather than a port of PhoenixOS's userspace API interception approach. The CRIU plugin is not equivalent to PhoenixOS and does not make PhoenixOS redundant for AMD.

**Decision**: Proceed with ROCm port -- genuine value-add. However, this is NOT a straightforward CUDA->HIP translation: the project intercepts and replays CUDA Driver/Runtime API calls, parses CUDA fatbins/PTX, and uses NVIDIA-specific tooling (`cuobjdump`, `fatbinary`, `cu++filt`). A full port requires implementing equivalent ROCm functionality.

## Build Classification

**Build system**: Meson (not CMake, not pytorch extension)

**Evidence**:
- Root `meson.build` (358 lines) is the primary build file
- No CMakeLists.txt at root
- No setup.py or pytorch extension usage
- `meson.build` line 1: `project('PhoenixOS', ['CPP'], ...)`

**Strategy**: Neither Strategy A (pure CMake) nor Strategy B (pytorch extension) applies. This requires a **new Strategy C: Meson-based project with platform-specific implementation modules**.

The project has an explicit platform abstraction designed for multiple backends:
- `pos/cuda_impl/` -- CUDA-specific implementation
- The architecture expects `pos/rocm_impl/` (or similar) to be created
- `meson.build` already has conditional compilation based on `conf_runtime_target`

## Port Strategy

**Approach**: Create a parallel ROCm implementation (`pos/rocm_impl/`) mirroring the CUDA implementation structure.

This is fundamentally different from a typical CUDA->HIP port because:

1. **Not kernel translation**: PhoenixOS does not contain CUDA kernels to be translated. The `.cu` files in `microbench/` and `utils/cudam/` are test utilities, not the core system.

2. **API interception layer**: The core is a hijacker (`libphos.so`) that intercepts CUDA Driver/Runtime API calls, records them, and replays them for checkpoint/restore. A ROCm port requires:
   - Intercepting HIP runtime calls (`hipMalloc`, `hipMemcpy`, `hipLaunchKernel`, etc.)
   - Intercepting HIP driver calls via the HIP Driver API
   - Parsing HIP fatbins (code objects) instead of CUDA fatbins
   - ROCm-specific memory management (HSA, KFD)

3. **Binary patching**: The Rust patcher (`pos/cuda_impl/patcher/`) patches PTX using `cuobjdump` and `fatbinary`. ROCm requires:
   - Parsing AMDGCN code objects (different format)
   - Using `roc-obj-ls`, `roc-obj-extract` instead of `cuobjdump`
   - Using `clang-offload-bundler` instead of `fatbinary`

4. **cuDNN -> MIOpen**: The API index includes cuDNN calls (5000-5319) which must map to MIOpen.

5. **NVML -> rocm-smi**: NVML calls (4000-4003) must use rocm-smi or AMD equivalent.

## CUDA Surface Inventory

### API Interception Scope (from api_index.h)

**CUDA Runtime APIs** (prefixed CUDA_): ~80 APIs
- Device management: CUDA_GET_DEVICE, CUDA_SET_DEVICE, CUDA_GET_DEVICE_PROPERTIES, etc.
- Memory: CUDA_MALLOC, CUDA_FREE, CUDA_MEMCPY_*, CUDA_MEMSET_*, CUDA_HOST_ALLOC
- Streams/Events: CUDA_STREAM_*, CUDA_EVENT_*
- Kernel launch: CUDA_LAUNCH_KERNEL, CUDA_LAUNCH_COOPERATIVE_KERNEL
- P2P: CUDA_DEVICE_CAN_ACCESS_PEER, CUDA_DEVICE_ENABLE_PEER_ACCESS

**CUDA Driver APIs** (prefixed rpc_cu): ~25 APIs
- cuInit, cuDeviceGet*, cuCtxGetCurrent, cuCtxSetCurrent
- cuModuleLoad, cuModuleLoadData, cuModuleGetFunction
- cuMemAlloc, cuMemcpyHtoD
- cuLaunchKernel

**cuBLAS APIs** (prefixed rpc_cublas): ~12 APIs
- cublasCreate, cublasDestroy, cublasSetStream
- cublasSgemm, cublasDgemm, cublasSgemv, cublasDgemv
- cublasSgemmEx, cublasSgemmStridedBatched

**cuDNN APIs** (prefixed rpc_cudnn): ~70 APIs
- Full CNN layer support: tensors, filters, convolution, pooling, activation, batch norm, LRN, softmax

**NVML APIs** (prefixed rpc_nvml): 4 APIs
- nvmlInit, nvmlShutdown, nvmlDeviceGetCount

**cuSolver APIs** (prefixed rpc_cusolverDn): 6 APIs
- cusolverDnCreate, cusolverDnDestroy, cusolverDnSetStream
- cusolverDnDgetrf, cusolverDnDgetrs, cusolverDnDgetrf_bufferSize

### Binary Format Handling

**Fatbin parsing** (`pos/cuda_impl/utils/fatbin.h`):
- Parses CUDA fatbin ELF structure (FATBIN_STRUCT_MAGIC 0x466243b1, FATBIN_TEXT_MAGIC 0xBA55ED50)
- Extracts kernel metadata from `.nv.info` sections
- Decompresses compressed fatbins
- Uses libelf for ELF parsing (compatible with ROCm)

**PTX patching** (`pos/cuda_impl/patcher/src/fatbin.rs`):
- Uses `cuobjdump -all -xptx` to dump PTX
- Parses and patches PTX
- Uses `fatbinary --create` to repack

### Handle Types

From `pos/cuda_impl/handle/`:
- CUcontext, CUmodule, CUfunction
- cudaStream_t, cudaEvent_t
- CUdeviceptr (device memory)
- cuBLAS handles

## Risk List

### Critical (Blocking)

1. **NVIDIA-specific binary format parsing**: The fatbin parser (`POSUtil_CUDA_Fatbin`) and kernel parser (`POSUtil_CUDA_Kernel_Parser`) are deeply tied to NVIDIA's fatbin/ELF structure. ROCm code objects use a different format (AMDGCN ELF with different section names like `.note.AMD*`, `.rodata`, `.text`).

2. **PTX patching with NVIDIA tools**: The Rust patcher (`fatbin.rs`) calls:
   - `cuobjdump` -- NVIDIA tool, no direct ROCm equivalent
   - `fatbinary` -- NVIDIA tool for creating fatbins
   - `cu++filt` for demangling (ROCm: use standard `c++filt` or `llvm-cxxfilt`)

3. **CUDA Driver API semantics**: PhoenixOS tracks CUmodule/CUfunction relationships and uses Driver API for restore (`cuModuleLoadData`, `cuModuleGetFunction`). HIP Driver API (`hipModuleLoad`, etc.) has similar semantics but different details.

4. **cuDNN -> MIOpen API mapping**: 70+ cuDNN APIs must map to MIOpen. APIs are NOT 1:1 (different descriptor types, algorithm selection, workspace management).

### High

5. **Kernel parameter introspection**: The system parses kernel signatures from fatbin metadata to determine pointer parameters and their directions (input/output/inout). ROCm code objects expose kernel metadata differently (via kernel descriptors, not `.nv.info` sections).

6. **Memory model differences**: CUDA/HIP have different managed memory semantics. The checkpoint/restore of `hipMallocManaged` memory may behave differently from `cudaMallocManaged`.

7. **Context management**: CUDA's primary context model vs HIP's context handling needs review. PhoenixOS uses `cuCtxCreate`, `cuCtxSetCurrent`, `cuCtxPushCurrent` extensively.

### Medium

8. **Build system (Meson)**: Adding ROCm support requires modifying `meson.build` to detect ROCm, find HIP libraries via pkg-config or CMake find scripts, and conditionally compile `pos/rocm_impl/`.

9. **Third-party dependencies**: The patcher uses Rust with `cxx` interop. The Rust code must be modified for ROCm code object handling.

10. **NVML -> rocm-smi/rsmi**: Device enumeration and management APIs differ. rocm-smi provides similar functionality but with different API.

11. **cu++filt -> c++filt**: Symbol demangling. `cu++filt` is CUDA-specific; standard `c++filt` or `llvm-cxxfilt` work for HIP symbols.

### Low

12. **libclang usage**: The project uses libclang for C++ parsing. This is cross-platform and should work on ROCm.

13. **libelf usage**: ELF parsing via libelf is platform-agnostic.

## File-by-File Change List

### New Files to Create

```
pos/rocm_impl/
├── api_context.h           # HIP API context definitions (mirror cuda_impl)
├── api_index.h             # HIP API index defines (hip_*, rocblas_*, miopen_*, etc.)
├── client.h                # ROCm client implementation
├── handle.h                # HIP handle types
├── parser.h                # HIP API parsers
├── worker.h                # HIP worker implementation
├── workspace.h             # ROCm workspace
├── handle/
│   ├── context.h/cpp       # hipCtx_t handle
│   ├── device.h/cpp        # HIP device handle
│   ├── event.h/cpp         # hipEvent_t handle
│   ├── function.h/cpp      # hipFunction_t handle
│   ├── memory.h/cpp        # Device memory handle
│   ├── module.h/cpp        # hipModule_t handle
│   ├── stream.h/cpp        # hipStream_t handle
│   └── rocblas.h/cpp       # rocBLAS handle (if cuBLAS supported)
├── src/
│   ├── api_context.cpp
│   ├── client.cpp
│   ├── handle.cpp
│   ├── worker.cpp
│   ├── workspace.cpp
│   ├── handle/*.cpp
│   ├── parser/
│   │   ├── hip_runtime.cpp
│   │   ├── hip_driver.cpp
│   │   └── rocblas.cpp
│   └── worker/
│       ├── hip_runtime.cpp
│       ├── hip_driver.cpp
│       └── rocblas.cpp
├── utils/
│   └── codeobj.h           # ROCm code object parsing (replaces fatbin.h)
├── patcher/                # ROCm code object patcher
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs
│   │   └── codeobj.rs      # AMDGCN code object handling
│   └── cxx/
└── proto/
    └── *.proto             # Protobuf for ROCm handle serialization
```

### Files to Modify

1. **meson.build** (root):
   - Add `rocm` as valid `conf_runtime_target` option
   - Add ROCm library detection (hip, rocblas, miopen, etc.)
   - Add `pos/rocm_impl/` sources when target is rocm
   - Link HIP runtime (`-lamdhip64`)

2. **unittest/meson.build**:
   - Support ROCm test target
   - Add `unittest/test_rocm/` sources

3. **scripts/build_scripts/build_configs.yaml**:
   - Add `rocm` target configuration
   - Add ROCm version configuration

4. **autogen/** (if code generation is used for ROCm):
   - Create `autogen_rocm/` directory
   - Generate ROCm parser/worker stubs

### Files to Leave Unchanged

- `pos/cuda_impl/*` -- Keep CUDA implementation intact
- `microbench/*` -- CUDA-only benchmarks, separate effort
- `examples/*` -- CUDA-only examples
- `third_party/*` -- Dependencies are platform-agnostic

## Build Commands

### Prerequisites

```bash
# ROCm installation (assumed 6.0+ for code object compatibility)
# apt install rocm-dev hip-runtime-amd rocblas miopen-hip

# Build dependencies
apt-get update
apt-get install git wget meson ninja-build pkg-config
apt-get install libprotobuf-dev protobuf-compiler
apt-get install libyaml-cpp-dev libelf-dev libibverbs-dev libuuid-dev
apt-get install clang libclang-dev   # For libclang usage

# Rust (for patcher)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Configure for ROCm (gfx90a)

```bash
cd PhoenixOS

# Set environment for ROCm build
export POS_BUILD_CONF_PlatformProjectRoot=$(pwd)
export POS_BUILD_CONF_RuntimeTarget=rocm
export POS_BUILD_CONF_RuntimeTargetVersion=6.0
export POS_BUILD_CONF_RuntimeEnablePrintError=1
export POS_BUILD_CONF_RuntimeEnablePrintWarn=1
export POS_BUILD_CONF_RuntimeEnablePrintLog=1
export POS_BUILD_CONF_RuntimeEnablePrintDebug=0
export POS_BUILD_CONF_RuntimeEnablePrintWithColor=1
export POS_BUILD_CONF_RuntimeEnableDebugCheck=1
export POS_BUILD_CONF_RuntimeEnableHijackApiCheck=0
export POS_BUILD_CONF_RuntimeEnableTrace=1
export POS_BUILD_CONF_RuntimeEnableMemoryTrace=0
export POS_BUILD_CONF_RuntimeDefaultDaemonLogPath=/var/log/phos/daemon
export POS_BUILD_CONF_RuntimeDefaultClientLogPath=/var/log/phos/client
export POS_BUILD_CONF_EvalCkptOptLevel=2
export POS_BUILD_CONF_EvalCkptEnableIncremental=1
export POS_BUILD_CONF_EvalCkptEnablePipeline=0
export POS_BUILD_CONF_EvalCkptDefaultIntervalMs=6000
export POS_BUILD_CONF_EvalMigrOptLevel=0
export POS_BUILD_CONF_EvalRstEnableContextPool=1

# Configure with meson
meson setup build --buildtype=release

# Build
ninja -C build
```

### Install

```bash
# Install phosd and libphos.so
ninja -C build install

# Or manual installation
cp build/libpos.so /usr/local/lib/libphos.so
```

## Test Plan

### Unit Tests (GoogleTest)

The test infrastructure exists but requires ROCm test cases:

```bash
# Build tests
cd unittest
meson setup build_test
ninja -C build_test

# Run tests
./build_test/pos_test
```

Expected test coverage (to be implemented):
- HIP memory allocation/free
- HIP stream/event operations
- HIP kernel launch and parameter capture
- rocBLAS handle creation and operations
- Checkpoint/restore cycle for HIP applications

### Integration Tests

1. **Basic checkpoint/restore**:
   ```bash
   # Start phosd
   pos_cli --start --target daemon
   
   # Run a simple HIP application with PhOS
   env $phos ./simple_hip_app
   
   # Checkpoint
   pos_cli --dump --dir /tmp/ckpt --pid <pid>
   
   # Restore
   pos_cli --restore --dir /tmp/ckpt
   ```

2. **Example applications** (need ROCm versions):
   - Simple vector add
   - Matrix multiplication (rocBLAS)
   - CNN inference (MIOpen)

### Non-GPU Tests

The following must not regress:
- Meson build system configuration parsing
- Protobuf serialization/deserialization
- YAML configuration loading
- ELF parsing utilities (libelf-based)
- CLI argument parsing

## Open Questions

1. **Scope of initial port**: Should the initial port support the full API surface (cuDNN/MIOpen, all cuBLAS/rocBLAS operations) or start with a minimal subset (basic memory, streams, kernel launch)?

2. **Code object patching**: Is PTX-level patching essential for checkpoint/restore, or can we operate on AMDGCN directly? The patching appears to instrument kernels for memory access tracking.

3. **Upstream appetite**: Given the project states ROCm is "Developing", should we coordinate with upstream maintainers before proceeding? They may have internal work in progress.

4. **CRIU integration**: Should the ROCm port leverage any functionality from CRIU's AMD GPU plugin, or remain independent?

5. **Library coverage priority**: Which libraries are highest priority?
   - Core: HIP runtime + driver API (essential)
   - High: rocBLAS (common in ML workloads)
   - Medium: MIOpen (cuDNN equivalent, complex API)
   - Low: hipFFT, hipSOLVER, etc.

6. **Memory model differences**: How does HIP managed memory behavior differ from CUDA, and what changes are needed for checkpoint/restore?

## Recommendation

**This project requires significant implementation effort beyond typical CUDA->HIP porting**. The port is essentially implementing a new platform backend from scratch, using the CUDA implementation as a reference.

**Suggested approach**:
1. Start with a minimal viable port: HIP runtime APIs only (malloc, memcpy, stream, event, kernel launch)
2. Implement code object parsing without binary patching initially
3. Add rocBLAS support
4. Defer MIOpen support (complex, lower priority for basic C/R)
5. Coordinate with upstream maintainers

**Effort estimate**: Large (weeks to months), not a typical few-hour port.
