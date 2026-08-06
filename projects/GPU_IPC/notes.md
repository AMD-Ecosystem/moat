# GPU_IPC notes

## Port Status: Blocked (Eigen device eigensolver incompatibility)

### Changes Made
- Created `cuda_to_hip.h` compatibility header with CUDA->HIP API mappings
- Modified CMakeLists.txt to support HIP build with `-DUSE_HIP=ON`
- Updated all headers to use cuda_to_hip.h instead of cuda_runtime.h
- Fixed kernel launch syntax spacing (`<< <` -> `<<<`, `>> >` -> `>>>`) in all .cu files
- Added missing return statements in mlbvh.cu and femEnergy.cu functions
- Enabled relocatable device code (-fgpu-rdc) for cross-TU device linking
- Added `__HIP_PLATFORM_AMD__` define for CXX files and ROCm include path

### Build Configuration
```bash
cmake .. -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_BUILD_TYPE=Release
```

### Blocking Issue
The project uses Eigen's `SelfAdjointEigenSolver` in device (`__device__`) code:
- `PDSNK<double, 12>()` in femEnergy.cu:855 -- used by bending gradient/Hessian kernel
- `__project_StabbleNHK_H_3D_makePD()` in femEnergy.cu:966 -- used by FEM kernel

Eigen's eigensolvers are not compatible with HIP device compilation because:
1. They allocate memory dynamically (not allowed in device code)
2. They use control flow and function calls that are host-only
3. HIP/clang does not have nvcc's special Eigen handling

NVCC has special support for Eigen that allows some host-only operations to work in `__device__` code through implicit compilation tricks. HIP/clang does not have this capability.

### Resolution Options
1. **Rewrite GPU eigensolver**: Implement a GPU-compatible symmetric eigenvalue solver (e.g., Jacobi method, bisection) to replace Eigen's SelfAdjointEigenSolver
2. **Use rocSOLVER**: Call hipSOLVER/rocSOLVER for eigendecomposition, but this requires significant kernel restructuring since the eigensolver is called per-element inside kernels
3. **Move to host**: Restructure the code to compute eigendecompositions on the host and pass results to GPU, but this would significantly impact performance

### Technical Details
The PDSNK function performs positive semi-definite projection on Hessian matrices, clamping negative eigenvalues to zero. This is used in:
- Bending energy gradient/Hessian computation (12x12 matrices)
- Stable Neo-Hookean material model (9x9 matrices)

These are called per-element inside CUDA kernels, making host-side computation impractical without major architectural changes.

### Files Modified
- GPU_IPC/cuda_to_hip.h (new)
- GPU_IPC/cuda_tools.h
- GPU_IPC/device_utils.h
- GPU_IPC/PCG_SOLVER.cuh
- GPU_IPC/load_mesh.h
- GPU_IPC/ACCD.cuh
- GPU_IPC/femEnergy.cuh
- GPU_IPC/gpu_eigen_libs.cuh
- GPU_IPC/mlbvh.cuh
- GPU_IPC/FrictionUtils.cuh
- GPU_IPC/GIPC_PDerivative.cuh
- GPU_IPC/gl_main.cpp
- GPU_IPC/MASPreconditioner.cu
- GPU_IPC/PCG_SOLVER.cu
- GPU_IPC/mlbvh.cu
- GPU_IPC/femEnergy.cu
- All .cu files: fixed kernel launch syntax spacing
- CMakeLists.txt
