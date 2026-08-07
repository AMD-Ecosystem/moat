# TornadoVM notes

## Intake screen (2026-08-07)

TornadoVM (beehive-lab/TornadoVM) is a JIT compiler that translates Java bytecode to run on accelerators. It has four backends: OpenCL C, NVIDIA CUDA (PTX), Apple Metal (MSL), and a multi-core CPU backend. OpenCL is the default backend and is documented as targeting NVIDIA, AMD and Intel GPUs directly (README.md: "OpenCL (default) | NVIDIA / AMD / Intel GPUs, multi-core CPUs, FPGAs"; also "dedicated GPUs from NVIDIA, AMD, and Intel"). AMD hardware is already reachable today through this backend without any ROCm/HIP work.

The CUDA-specific surface is not CUDA kernels in-tree: repo-wide there are only 3 `.cu`/`.cuh` files, one of which is a generated example output and one is a CUTLASS JNI wrapper (`tornado-drivers/cutlass-jni/.../tornado-cutlass.cu`). The `tornado-drivers/ptx` module is TornadoVM's own JIT compiler backend: it emits PTX assembly text from its internal IR and loads it via the NVIDIA driver/NVRTC APIs -- CUDA is a compilation TARGET for this backend's code generator, not a body of CUDA source to hipify. The newer `tornado-cublas`/`tornado-cudnn`/`tornado-cufft`/`tornado-cusparse` modules (mostly README/pom.xml/ROADMAP stubs right now) are direct JNI bindings to NVIDIA's proprietary math libraries and Tensor Core `mma.sync` intrinsics -- an AMD analogue would mean new bindings against rocBLAS/MIOpen/hipFFT/hipSPARSE plus a from-scratch AMDGPU code-generation backend in the compiler, not a source port. This matches the reimplement-not-port pattern already dispositioned for mirage, SpargeAttn and FlashRT.

### Licence

Dual/tri-licensed by module, documented explicitly in README.md ("Licenses per module") and confirmed against the LICENSE files in the tree (`LICENSE_APACHE2`, `LICENSE_GPLv2CE`, `LICENSE_MIT`):
- Tornado-API, Tornado-Assembly, Tornado-scripts, Tornado-Annotation, Tornado-Unittests, Tornado-Benchmarks, Tornado-Examples, Tornado-Matrices, Tornado-Drivers-OpenCL-Headers: Apache-2.0 (tier 1).
- Tornado-Runtime, Tornado-Drivers (this is where any CUDA/PTX-backend work would land): GPL-2.0 with Classpath Exception (base SPDX GPL-2.0-only, tier 2 -- "approved for contributing to third-party projects").

Recording `license_spdx: "Apache-2.0 OR GPL-2.0-only WITH Classpath-exception-2.0 OR MIT"` (per-module, not ambiguous -- the split is explicit and documented, not a guess). Both applicable tiers (1 and 2) are cleared to contribute; this is not a gating concern for either fork or decline.

### Duplicate effort / viability verdict: DECLINE, already-supported

AMD GPUs are already reachable through TornadoVM's own default OpenCL backend -- this is not a partial/parked port by another team, it is upstream's own shipped functionality, so `already-supported` is the closer fit than `not-a-target` even though the CUDA-only slice (PTX backend, CUTLASS/cuBLAS/cuDNN/cuFFT/cuSPARSE JNI bindings) is also, independently, a code-generation/library-binding target rather than in-tree CUDA kernels -- the same reimplement-not-port shape as mirage/SpargeAttn/FlashRT. Recommending decline on both grounds: the goal (Java code running on AMD GPUs) is already met without any port, and the part that is CUDA-specific would not be a port even if pursued.

Upstream is active (not archived); no dependency on any other MOAT project.
