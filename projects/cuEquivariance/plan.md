# cuEquivariance Port Plan

## Project

- **Name**: cuEquivariance
- **Upstream**: https://github.com/NVIDIA/cuEquivariance
- **Default branch**: main

## Existing AMD support

**None.** No ROCm, HIP, or AMD GPU support exists anywhere:
- Upstream repo: no AMD/ROCm references in docs or code
- GitHub forks: ~20 forks, none with AMD/ROCm work
- Web search: no community ports, no AMD documentation
- PyPI: ops packages are binary-only for CUDA 12/13, no AMD variants

## Build classification

**Not applicable.** This is NOT a standard CUDA->HIP port target.

The repository structure:
- `cuequivariance/` -- pure Python, math descriptors and segment specifications
- `cuequivariance_torch/` -- PyTorch frontend (pure Python, imports `cuequivariance_ops_torch`)
- `cuequivariance_jax/` -- JAX frontend (pure Python, imports `cuequivariance_ops_jax`)

There are **zero `.cu`, `.cuh`, `.cpp`, or `.c` files** in this repository.

## Evidence: CUDA kernels are closed-source binaries

The performance-critical CUDA kernels are distributed as **closed-source binary wheels**:
- `cuequivariance-ops-torch-cu12` / `cuequivariance-ops-torch-cu13`
- `cuequivariance-ops-jax-cu12` / `cuequivariance-ops-jax-cu13`

From PyPI ([cuequivariance-ops-cu12](https://pypi.org/project/cuequivariance-ops-cu12/)):
> "cuequivariance_ops provides CUDA kernels for cuEquivariance. This python module does not contain any python bindings. When imported, it loads the shared library that contains the CUDA kernels."
>
> **Source Distributions: "No source distribution files available for this release."**

The `cuequivariance_torch/__init__.py` imports from `cuequivariance_ops_torch` (lines 64, 77), confirming the kernel code lives in a separate, non-public package.

## Port strategy

**Not applicable.** MOAT's Strategy A (CMake compat-header) and Strategy B (torch-hipify) both require CUDA source code to translate. No such source exists in this repository.

## CUDA surface inventory

**Empty.** No CUDA code to inventory.

## Risk list

1. **No source to port** -- the entire GPU acceleration surface is closed-source binary
2. **Greenfield scope** -- a ROCm backend would require reimplementing the kernels from:
   - The mathematical specifications in the open Python frontend
   - The segmented tensor product / equivariant polynomial descriptions
   - Potentially reverse-engineering behavior from the CUDA binaries
3. **Performance parity** -- NVIDIA's kernels are tuned for their hardware; matching performance on AMD requires AMD-native optimization (rocWMMA, Composable Kernel, MFMA), not just translation
4. **NVIDIA project** -- upstream PR not applicable (standing rule); outcome would be a standalone ROCm implementation

## File-by-file change list

N/A -- no files to modify.

## Build commands

N/A -- no CUDA source to build.

## Test plan

N/A -- no port to test.

## Open questions

1. **Pursue greenfield or block?** A ROCm backend for cuEquivariance would be a significant engineering project (writing the segmented polynomial kernels from scratch for AMD GPUs). This is research-grade scope, similar to the greenfield hipCIM/hipML class of efforts. Decision needed: is this strategic enough to justify the investment, or should we block as out-of-scope?

## Disposition

**BLOCKED / NOT A TRANSLATION PORT TARGET**

The open-source repository contains only Python frontend code. The CUDA kernels that would need to be ported are distributed as closed-source binary wheels by NVIDIA. There is nothing to hipify.

If an AMD implementation is desired, it would be a **greenfield reimplementation** of the kernel functionality, not a MOAT-style CUDA->HIP translation. Such work would live as a separate project (e.g., a ROCm backend plugin for the open cuequivariance frontend), not as a fork of NVIDIA/cuEquivariance.

Recommended disposition: `skip` with reason `cant-port` -- closed-source binary kernels.
