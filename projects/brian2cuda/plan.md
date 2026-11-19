# Brian2CUDA Port Plan (linux-gfx90a)

## Project

- Name: brian2cuda
- Upstream: https://github.com/brian-team/brian2cuda
- Default branch: main
- Description: Brian2CUDA is a Python package that extends the Brian2 spiking neural network simulator to generate and run CUDA code on NVIDIA GPUs. It is a code generator, not a precompiled CUDA library.

## Existing AMD support

**Decision: Port from scratch (no existing AMD support)**

Searches performed:
- `grep -rniE 'amd|rocm|hip|gfx[0-9]' README* docs/`: No AMD/ROCm mentions found.
- WebSearch "brian2cuda ROCm", "brian2cuda AMD GPU", "brian2cuda HIP": No results for brian2cuda+AMD/HIP.
- `gh api repos/brian-team/brian2cuda/forks`: 20+ forks examined, none with rocm/hip/amd in name or description.
- No upstream rocm/hip branches or PRs.

The project has open issues/PRs for OpenCL (#7) and Metal (#6) backends, but no HIP backend effort exists.

**Merge policy**: The project is maintained by the Brian team and appears to accept platform contributions (given the OpenCL/Metal PRs). An upstream PR for a HIP backend would likely be welcomed.

## Build classification

**Classification: Code generator with runtime compilation (neither Strategy A nor Strategy B)**

Evidence:
- pyproject.toml: Pure Python package installed via pip, depends on brian2
- No CMakeLists.txt at all
- No precompiled `.cu` files to hipify
- `.cu` files in `templates/` are Jinja2 templates, not compilable sources
- `device.py` (106KB) implements code generation and runtime nvcc compilation
- `cuda_generator.py` (46KB) translates Brian2 models to CUDA C++ code

This is a **runtime code generator**: Python code generates C++/CUDA source code from neural network model descriptions, then compiles it with nvcc at runtime. There is no static CUDA codebase to port.

## Port strategy

**Strategy: New HIP backend (parallel to CUDA backend)**

This is a fundamentally different port from Strategy A or B. The work involves:

1. **Creating a new `hip_standalone` device** parallel to `cuda_standalone`
2. **Duplicating and modifying the code generator** to emit HIP-compatible code
3. **Creating HIP-specific templates** (20+ template files) or a HIP compat layer
4. **Replacing cuRAND with hipRAND** in the code generator
5. **Generating hipcc Makefiles** instead of nvcc Makefiles
6. **Adding HIP device detection and selection**

The architecture favors a clean HIP backend rather than trying to make the CUDA backend emit HIP-compatible code. This is because:
- The code generator deeply knows about CUDA-specific APIs (curand states, nvcc flags)
- Template files reference CUDA-specific headers and APIs
- A clean separation makes maintenance easier and avoids NVIDIA/AMD conditionals everywhere

### Scope estimation

The port is substantial:
- `device.py`: ~2800 lines - core device abstraction, needs HIP version
- `cuda_generator.py`: ~1200 lines - code generation, needs HIP adaptation
- `cuda_prefs.py`: ~350 lines - CUDA preferences, needs HIP version
- Templates: ~20 files totaling ~2500 lines - need HIP versions or compat layer
- `brianlib/`: ~8 header files - need HIP versions

Total: ~6000+ lines of code to duplicate/adapt plus significant architectural work.

### Recommended approach

Given the scope, a phased approach is recommended:

**Phase 1: Minimal viable HIP backend**
- Create `hip_standalone` device with basic functionality
- Add HIP compat header to templates (CUDA->HIP symbol mapping)
- Modify Makefile generator to use hipcc
- Get basic neuron group simulations running

**Phase 2: Full feature parity**
- hipRAND integration (device and host API)
- All template files working with HIP
- Full test suite passing

**Phase 3: Upstream preparation**
- Code cleanup and documentation
- Windows HIP support (if feasible)
- Upstream PR

## CUDA surface inventory

### Code generator files (Python - generate CUDA code)
| File | Lines | Description |
|------|-------|-------------|
| device.py | ~2800 | Main device class, generates CUDA project, compiles with nvcc |
| cuda_generator.py | ~1200 | Translates Brian2 to CUDA code, generates atomic operations |
| cuda_prefs.py | ~350 | CUDA-specific preferences (GPU selection, nvcc flags) |
| codeobject.py | ~130 | Code object classes |
| binomial.py | ~100 | Binomial random number generation (uses curand) |

### Template files (Jinja2 templates, generate .cu files)
| Template | Purpose |
|----------|---------|
| main.cu | Main program entry point |
| objects.cu | Global objects initialization, queries `props.warpSize` |
| run.cu | Simulation run control |
| stateupdate.cu | Neuron state update kernels |
| threshold.cu | Spike threshold detection |
| reset.cu | Post-spike reset |
| synapses.cu | Synaptic propagation |
| synapses_classes.cu | Synapse class definitions |
| synapses_create_*.cu | Synapse generation |
| synapses_push_spikes.cu | Spike queue management |
| spikemonitor.cu | Spike monitoring (uses Thrust) |
| statemonitor.cu | State variable monitoring |
| ratemonitor.cu | Rate monitoring |
| rand.cu | Random number generation (curand host API) |
| network.cu | Network management |
| common_group.cu | Common group operations |
| summed_variable.cu | Summed variable computation |
| spatialstateupdate.cu | Spatial state updates |

### Runtime headers (C++)
| Header | Purpose | HIP porting notes |
|--------|---------|-------------------|
| cuda_utils.h | Error checking macros | Map to HIP |
| curand_buffer.h | Random number buffer | Replace with hiprand |
| cudaVector.h | Device vector utilities | Use HIP versions |
| spikequeue.h | Spike queue implementation | Should work with HIP |
| dynamic_array.h | Dynamic array utilities | Should work with HIP |
| clocks.h | Clock utilities | Host-only, no changes |
| common_math.h | Math utilities | Should work with HIP |

### CUDA library usage
| Library | Usage | HIP equivalent |
|---------|-------|----------------|
| cuRAND | Random number generation (device + host API) | hipRAND |
| Thrust | sort, device_ptr, raw_pointer_cast | rocThrust |
| CUDA Runtime | Memory management, kernel launch | HIP Runtime |

### CUDA features used in templates
- `cudaMalloc`, `cudaFree`, `cudaMemcpy` - standard runtime
- `cudaSetDevice`, `cudaGetDeviceProperties` - device management
- `cudaDeviceSynchronize`, `cudaStreamSynchronize` - synchronization
- `cudaDeviceSetLimit` - heap size configuration
- `curandCreateGenerator`, `curandGenerate*` - random numbers
- `curandState`, `curand_uniform`, `curand_normal` - device-side RNG
- `thrust::device_ptr`, `thrust::raw_pointer_cast`, `thrust::sort` - Thrust
- `atomicAdd`, `atomicCAS` - atomic operations
- `props.warpSize` (runtime query) - warp size

### Warp size handling
The project correctly queries warp size at runtime via `props.warpSize` (objects.cu:362), which is the correct approach for wave64/wave32 compatibility. No hardcoded `32` for warp size found.

## Risk list

| Risk | Severity | Mitigation |
|------|----------|------------|
| Large scope (6000+ lines) | High | Phased approach, prioritize core functionality |
| cuRAND device API | Medium | hipRAND has similar API, careful mapping needed |
| Thrust dependencies | Low | rocThrust is drop-in compatible |
| nvcc-specific flags in generator | Medium | Abstract compiler interface |
| Template complexity | Medium | Consider HIP compat header for templates |
| Test suite requires GPU | Medium | Need real GPU validation |

## Build commands

### Install brian2cuda (development)
```bash
pip install -e .
```

### Run tests (requires GPU)
```bash
pytest brian2cuda/tests/ -v
```

### Example usage
```python
from brian2 import *
import brian2cuda
set_device("cuda_standalone")  # Would become "hip_standalone"

# Define neural network
G = NeuronGroup(100, 'dv/dt = -v/(10*ms) : 1')
run(1*second)
```

## Test plan

### GPU tests
The test suite is comprehensive (18 test files) covering:
- Code generation correctness
- Random number generation
- Synaptic propagation
- State updates
- Monitors

Run with: `pytest brian2cuda/tests/ -v`

### Validation approach
1. Run same neural network model on CPU (Brian2 cpp_standalone) and GPU (HIP backend)
2. Compare spike times, state variables within tolerance
3. Verify deterministic RNG produces same sequences as CUDA backend

### Non-GPU tests
- String manipulation utilities
- Code parsing tests
- GPU detection tests (mocked)

## Open questions

1. **Compat header vs full template rewrite**: Should templates include a HIP compat header for minimal changes, or should HIP-specific templates be created? The compat header approach is lower effort but may leave CUDA artifacts in the generated code. Full rewrite is cleaner but doubles template maintenance.

2. **Upstream preference**: Should the HIP backend be a separate package (`brian2hip`) or integrated into brian2cuda with a device switch? The upstream maintainers should be consulted.

3. **cuRAND state management**: The project uses cuRAND device-side API with per-thread state management. hipRAND has similar API but needs careful testing. Are there any cuRAND features not available in hipRAND?

4. **Thrust version compatibility**: The project uses Thrust features. Is rocThrust on ROCm 7.x fully compatible with the used features (device_ptr, raw_pointer_cast, sort)?

5. **Windows support**: The upstream explicitly does not support Windows (`raise RuntimeError("Windows is currently not supported")`). Should the HIP port target Linux only initially?

## Recommendation

This project is a significant undertaking due to its code-generator architecture. Unlike static CUDA codebases where hipify can do most of the work, here the Python code generator must be modified to emit HIP-compatible code.

However, the project:
- Uses warpSize correctly (runtime query)
- Has no complex warp intrinsics
- Uses standard CUDA libraries with HIP equivalents

A successful port would provide AMD GPU acceleration for the entire Brian2 neural simulator ecosystem, benefiting computational neuroscience researchers.

**Estimated effort**: High (2-4 weeks for Phase 1, 1-2 more weeks for full parity)
