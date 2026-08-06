# kaldi -- ROCm/HIP port plan (linux-gfx90a lead)

## Project
- Name: kaldi
- Upstream: https://github.com/kaldi-asr/kaldi
- Default branch: master
- Clone (read-only, gitignored): projects/kaldi/src/  (kaldi's own tree is the nested src/, i.e. projects/kaldi/src/src/)
- HEAD analyzed: e02e35f0254bb033fab73d1df99fc34123e31d56 (2025-09-22)

## Existing AMD support -> decision: FINISH/IMPROVE (bitrotted upstream HIP path), do NOT skip

kaldi already ships a mature, upstream-merged ROCm/HIP backend, authored by AMD's ROCm team
(`sfantao`), introduced 2022-09 and last touched 2023-11, written against ROCm 5.0.2-5.2.x.
It is broad and well-engineered, not a stub:

- `src/configure` has first-class `--use-rocm`, `--rocm-dir=DIR`, `--rocm-targets=TGTS`. `configure_rocm`
  writes `IS_GPU_BUILD=true`, `ROCM=true`, `HIPCC=$ROCMDIR/bin/hipcc`, one `--offload-arch=<tgt>` per
  requested target into `ROCM_ARCH_FLAGS`, reads `ROCM_MAJOR/MINOR_VERSION` from `hipconfig`, requires
  ROCm >= 5.2, and appends `src/makefiles/hip_64bit.mk` to `kaldi.mk` (configure:258-313).
- `src/makefiles/hip_64bit.mk` is a complete HIP toolchain block: `-D__IS_HIP_COMPILE__ -D__HIP_PLATFORM_AMD__`,
  `-DCUDA_VERSION=11000 -D__CUDACC_VER_MAJOR__=11`, `-munsafe-fp-atomics -fgpu-default-stream=per-thread`,
  include paths for hipsparse/hipfft/hipblas/hiprand/rocrand, and link line
  `-lhipblas -lhipsparse -lhipsolver -lhiprand -lhipfft -lroctx64 -lamdhip64 -Wl,--allow-shlib-undefined`.
- `src/hip/hipify.h` (283 lines) is a single colmap-style compat header aliasing every CUDA spelling the
  project uses (cublas*/cusparse*/cusolverDn*/curand*/cufft*/cuda* runtime + nvtx->roctx + `#define cub hipcub`),
  with thoughtful shims (a wave-correct `__syncwarp` = `s_nop`, a zero-size-guarded `cudaMemcpy2DAsync`,
  `cudaStreamPerThread`=(hipStream_t)2, `__fdiv_rd`->`__fdiv_rn`, `GPU_WARP_SIZE 64`).
- 52 source files participate (cudamatrix, cudafeat, cudadecoder, chain). Per-dir Makefiles compile `.cu`
  with `$(HIPCC) -c -x hip ... $(ROCM_ARCH_FLAGS)` under `ifeq ($(ROCM),true)` (cudamatrix/Makefile:37-40).

Per PORTING_GUIDE "an upstream HIP path bitrots against the ROCm in use; the MOAT value is to build +
GPU-validate it and fix the rot" (GPUMD, STRUMPACK lessons): this is a FINISH/IMPROVE target, not a skip.
The README/CI advertise no ROCm and there is no ROCm CI, so it has had no recent validation against
modern ROCm. The single confirmed blocker below proves it does not build on ROCm 7.2.1 as-is.

## Build classification: native Makefile/configure build (Strategy A bucket) -- ext_type=cmake

Evidence: the canonical kaldi build is `tools/` (OpenFST + a BLAS) then `src/configure` -> generates
`src/kaldi.mk` -> recursive GNU `make`. There is NO pytorch / `torch.utils.cpp_extension` / `find_package(Torch)`
anywhere. (A top-level `CMakeLists.txt` exists as an alternative, but the GPU/ROCm wiring lives ENTIRELY in
the configure+Makefile path; the CMake path does not implement `--use-rocm`.) This is the Strategy-A family
(pure native build, `.cu` translation units compiled by the GPU toolchain, host C++ untouched). `ext_type`
is recorded as `cmake` only because that is MOAT's enum for "non-pytorch native build, Strategy A"; the
actual mechanism is Makefile+configure, and the port must target that path, not CMake.

## Port strategy: Strategy A, already implemented -- fix bitrot only (minimal diff)

Strategy A (single compat header + `.cu` compiled `-x hip` + a few guards) is ALREADY in place and
correct in structure. The work is NOT a fresh conversion; it is: bring the existing HIP build green on
ROCm 7.2.1/gfx90a, GPU-validate the cudamatrix test suite, and improve only what the modern stack forces.
Keep the diff minimal and the CUDA (NVIDIA) path byte-for-byte unchanged. Make the arch configurable (it
already is, via `--rocm-targets`), so gfx1100/gfx1151 followers need only `--rocm-targets=gfx1100` and no
source edit.

## CUDA surface inventory (all already mapped in hipify.h; libs all present on ROCm 7.2.1)

- Kernels / device code: cudamatrix/cu-kernels.cu (largest), cudadecoder/cuda-decoder-kernels.cu,
  cudadecoder/batched-static-nnet3-kernels.cu, chain/chain-kernels.cu, and ~8 cudafeat/*-kernels.cu.
- Warp intrinsics: NO `__shfl*`/`__ballot`/`__activemask` in the kernels. Only `__syncwarp` (handled:
  hipify.h provides an `s_nop` wave-lockstep version; the in-file `#if CUDA_VERSION<9000` fallback def in
  cu-kernels.cu is disabled because hip_64bit.mk sets CUDA_VERSION=11000). Warp size is abstracted as
  `GPU_WARP_SIZE` = 64 on HIP (hipify.h:280), 32 on CUDA (cu-common.h:49) -- the canonical abstraction;
  host launch geometry derives block dims from it (cu-sparse-matrix.cc, cu-vector.cc, cudafeat kernels).
- Textures / surfaces: NONE. (The worst fault classes -- texture pitch, linear-filter, layered arrays --
  do not apply to kaldi.)
- Libraries: cuBLAS->hipBLAS, cuSPARSE->hipSPARSE (generic CreateCsr/SpMM + legacy Dcsr2csc),
  cuSOLVER(Dn Cholesky: Spotrf/Spotrs/_bufferSize + Batched)->hipSOLVER, cuRAND->hipRAND,
  cuFFT(R2C, PlanMany)->hipFFT, CUB->hipCUB (`#define cub hipcub`), NVTX->roctx. All aliased in hipify.h;
  all `.so` present (libhipblas/hipsparse/hipsolver/hiprand/hipfft/roctx64/amdhip64 under /opt/rocm/lib).
- `__CUDA_ARCH__`: each GPU `.cu` does `#define __CUDA_ARCH__ 800` at top (cu-kernels.cu:31,
  feature-online-cmvn-cuda.cu:19, chain-kernels.cu:24) so the `#if __CUDA_ARCH__>=600` device-intrinsic
  branches (e.g. native fp64 atomicAdd) activate under hipcc. This is the raft-style "define a high arch so
  intrinsic branches fire" pattern, done per-file (intentional, not a guard bug). Watch it during bringup
  but it is by design.
- Pinned/stream/event: cudaMallocHost->hipHostMalloc, cudaHostRegister, cudaStreamPerThread (=(hipStream_t)2),
  events, cudaMemcpy2DAsync (zero-size-guarded shim) -- all mapped.
- Atomics: kernels use atomicAdd (incl. fp64 via the CUDA_ARCH>=600 branch) and `-munsafe-fp-atomics`.
  No atomicMin/Max-on-managed and no sub-word atomic CAS surface seen (the cuCollections/cudaKDTree fault
  classes do not apply).

## Risk list (cite PORTING_GUIDE fault classes)

1. CONFIRMED BLOCKER -- C++14 vs rocPRIM/hipCUB C++17 (PORTING_GUIDE 2026-05-30 GPUMD lesson).
   `hip_64bit.mk` ROCM_FLAGS hardcodes `-std=c++14`. On ROCm 7.2.1, `rocprim/config.hpp:68-69`
   `#error "rocPRIM requires at least C++17"`, and hipCUB pulls rocPRIM. Verified empirically:
   `hipcc -x hip -std=c++14` on a 3-line `#include <hipcub/hipcub.hpp>` FAILS (exit 141, the #error);
   `-std=c++17` PASSES (exit 0). Every cub-using TU (all of cudamatrix + 6 cudafeat kernels) fails to
   compile today. Fix: bump the HIP build to `-std=c++17` in hip_64bit.mk ROCM_FLAGS (CUDA path untouched).
   This is almost certainly the primary thing standing between this port and a green build. Low risk:
   C++17 is a superset; only the HIP TUs change.

2. Warp size 32-vs-64 (PORTING_GUIDE warp-size fault class). gfx90a is wave64; GPU_WARP_SIZE=64 is correct
   for CDNA. Because the kernels use NO cross-lane shuffles/ballots (only block-wide reductions via
   __shared__ + cub::BlockReduce sized by GPU_WARP_SIZE, and __syncwarp as a fence), the wave64 algorithm
   hazards (warp-synchronous reduction races, ballot truncation) that bit other projects are UNLIKELY here.
   But the kernels were authored and last validated on wave64 ROCm 5.2 -- treat any cudamatrix test
   numerical failure as a candidate wave64 reduction-order issue and check the specific kernel
   (cu-kernels.cu reductions around the guarded __syncwarp sites: 973/1183/1228/1280/1364/1816/1961/2030/
   2071/2129/2994/3285). FOLLOWERS gfx1100/gfx1151 are wave32: GPU_WARP_SIZE on HIP is a hard 64 in
   hipify.h (NOT arch-conditional), so on RDNA the value is WRONG. The delta-plan for the followers must
   make GPU_WARP_SIZE per-arch (`__GFX9__`-keyed 64 else 32, per the PORTING_GUIDE device-warp-size recipe),
   or query at runtime for host launch geometry. Defer to the follower delta-plan; gfx90a is unaffected.

3. hipBLAS v2 enum / API drift (PORTING_GUIDE library-swap fault class). hipify.h maps cublas compute types
   to `HIPBLAS_R_32F` and `cublasComputeType_t`->`hipblasDatatype_t`. ROCm 7.x hipBLAS deprecated
   `hipblasDatatype_t` in favor of `hipDataType`/`hipblasComputeType_t` for GemmEx. The GemmEx/GemmBatchedEx
   paths (cublas-wrappers.h, used by cu-matrix) are the likeliest compile/deprecation-warning churn against
   7.2.1. Risk: medium (compile warnings or a signature mismatch on GemmEx). Resolve by updating only the
   affected hipify.h aliases; keep `--allow-shlib-undefined` out of the diagnosis.

4. hipFFT/hipRAND/hipSOLVER status-enum and signature coverage (PORTING_GUIDE MPPI/raft lessons). hipify.h
   already const_casts trsm operands and maps the enum set used. Low residual risk; watch hipsolverDnSpotrf
   bufferSize signature (hipSOLVER occasionally differs from cuSOLVER in workspace-size argument order).

5. roctx header path. cu-common.h includes `<roctracer/roctx.h>`; present on 7.2.1 at exactly that path.
   The link uses `-lroctx64`. OK. (ROCM_USEROCTX/-DUSE_NVTX is commented out by default; roctx ranges are
   still compiled because cu-common.h includes the header unconditionally under HIP -- fine.)

6. Deprecation-warning-as-error risk. The build does not pass -Werror, so hipBLAS/hipFFT deprecation
   warnings should not break it; confirm no `-Werror` leaks in via default_rules.mk during bringup.

No rule-of-five/texture-handle, no OOB-neighbor, no 256B-pitch, no layered-array, no sub-word-atomic,
no spin-lock/ITS risks were found in the surface scan (those fault classes do not apply to kaldi).

## File-by-file change list (expected; minimal-diff, fix-the-rot)

- src/makefiles/hip_64bit.mk -- change `-std=c++14` -> `-std=c++17` in ROCM_FLAGS (CONFIRMED necessary).
  Possibly also confirm CXXFLAGS host std matches (host C++ is built by the system CXX via kaldi.mk;
  kaldi already requires C++17-capable host compilers, so host side is unaffected).
- src/hip/hipify.h -- ONLY IF the hipBLAS GemmEx compute-type/datatype enums fail against 7.2.1: update the
  `CUBLAS_COMPUTE_32F*` / `cublasComputeType_t` / `CUDA_R_*`->`HIP_R_*` aliases to the current hipBLAS
  spelling. Touch nothing else.
- src/configure -- ONLY IF a ROCm 7 version check or `hipconfig -v` format needs adjusting (the >=5.2 gate
  is fine for 7.2.1). Expected: no change.
- Per-kernel `.cu` -- expected NO changes for gfx90a (no warp-intrinsic edits needed). Any edit here would
  be a genuine wave64 numerical fix surfaced by a failing cudamatrix test, guarded by `__IS_HIP_COMPILE__`.

Keep every change inside `#ifdef __IS_HIP_COMPILE__` / the ROCM make path so the NVIDIA build is identical.

## Build commands (gfx90a)

Prereqsts (host): a C++17 g++, plus kaldi's `tools/` (OpenFST + a BLAS). From projects/kaldi/src:
```
cd tools && extras/check_dependencies.sh   # report only
make -j$(nproc)                             # builds OpenFST, sph2pipe, etc. (CPU-only, no GPU)
```
Configure + build the ROCm GPU port (from projects/kaldi/src/src):
```
./configure --use-rocm --rocm-dir=/opt/rocm --rocm-targets=gfx90a --use-cuda=no
# (configure auto-detects /opt/rocm via hipcc/hipconfig; --rocm-targets sets --offload-arch=gfx90a)
make -j$(nproc) depend
make -j$(nproc)      # builds base..cudamatrix..cudafeat..cudadecoder..bins
```
The arch is configurable via `--rocm-targets`, so followers build with `--rocm-targets=gfx1100` (or
gfx1151) and NO source change -- design the lead bringup with that in mind (PORTING_GUIDE configurable-arch
lesson). Pass all testable arches at planning time where possible.
A CPU-only compile smoketest in `rocm/dev-ubuntu-24.04:7.2.4-complete` is acceptable as a manual compile
check only (never the validation gate; it observes no GPU fault).

## Test plan

Real GPU tests (the validator runs these on gfx90a; `make test` in a dir runs each TESTFILE binary and
checks exit status -- a nonzero exit = FAIL; many tests internally assert bitwise/tolerance correctness):

- PRIMARY: `cd src/cudamatrix && make test` -- 12 GPU unit-test binaries exercising the cudamatrix layer
  end-to-end on device (cu-vector-test, cu-matrix-test, cu-math-test, cu-test, cu-sp-matrix-test,
  cu-packed-matrix-test, cu-tp-matrix-test, cu-block-matrix-test, cu-sparse-matrix-test, cu-array-test,
  cu-device-test, cu-compressed-matrix-test). These cover hipBLAS (gemm/gemv/ger/syrk/trsm/dot/axpy/nrm2),
  hipSPARSE (csr SpMM, csr2csc), hipSOLVER (Cholesky), hipRAND, hipCUB reductions, and the bespoke kernels.
  Plus speed tests (cu-matrix-speed-test, cu-vector-speed-test, cu-sp-matrix-speed-test, cu-rand-speed-test)
  -- run for non-crash + timing, not correctness gates.
  Run SERIALLY on a single assigned GPU (PORTING_GUIDE MPPI lesson: `make test`, not parallel ctest).
- cudafeat / cudadecoder: NO unit TESTFILES (their Makefiles set `TESTFILES=` empty). They are validated by
  building their binaries (cudafeatbin, cudadecoderbin: e.g. batched-wav-nnet3-cuda) and, if time permits,
  running a small egs recipe's GPU feature/decoding step on a tiny shipped dataset (e.g. yesno or an
  egs/mini_librispeech feature-extraction step) to confirm the spectral/ivector kernels and the cuda
  decoder run on device and produce sane output. The decisive GPU correctness gate is the cudamatrix suite;
  cudafeat/cudadecoder are a build + smoke-run check above that.
- Determinism: where a cudamatrix test prints results, two runs should be bit-identical (the reductions are
  __syncthreads-tree / cub::BlockReduce, deterministic). A nondeterministic cudamatrix result fingerprints a
  wave64 reduction-order bug (PORTING_GUIDE MPPI/popsift) -- investigate the specific kernel.

Non-GPU regression set that must NOT regress: kaldi's CPU libraries and their tests build the same on a
ROCm-configured tree (the ROCm flags only affect cudamatrix/cudafeat/cudadecoder/chain). Sanity: build the
full tree and run a couple of CPU test dirs (e.g. `cd src/matrix && make test`, `cd src/base && make test`)
to confirm the configure/Makefile changes did not disturb the host build. The C++17 bump is HIP-only
(ROCM_FLAGS), so CPU TUs are unaffected.

## Inter-project MOAT deps

NONE. kaldi's GPU stack depends only on ROCm system libraries (hipBLAS/hipSPARSE/hipSOLVER/hipRAND/hipFFT/
hipCUB/roctx), all shipped with ROCm 7.2.1, plus OpenFST + a BLAS built locally in tools/. No RAPIDS/rmm/
raft/cuco/CUTLASS or other MOAT-ported project is in the dependency closure. `set-deps` -> none.

## Open questions

- Does the cudamatrix suite pass numerically on wave64/ROCm 7.2.1 after the C++17 bump, or does a specific
  reduction kernel need a wave64 fix? (Resolve by running the suite; the surface scan suggests it will pass
  because there are no cross-lane warp ops, but the code was last validated on ROCm 5.2.)
- Does hipBLAS GemmEx on 7.2.1 still accept the `hipblasDatatype_t`/`HIPBLAS_R_32F` spellings in hipify.h,
  or must those aliases move to `hipDataType`/`hipblasComputeType_t`? (Resolve at first compile of
  cu-matrix.)
- Validator scope for cudafeat/cudadecoder beyond compile: which shipped egs recipe (yesno is smallest) can
  exercise the cuda feature + decoder kernels quickly on device without large model/data downloads?

## Delta plan: followers (to be filled on demand)
gfx1100/gfx1151 are RDNA wave32. The single known arch delta is GPU_WARP_SIZE: hipify.h hardcodes it to 64,
which is wrong on RDNA. Make it per-arch (`__GFX9__` -> 64 else 32) for device code and/or query warpSize at
runtime for host launch geometry, then re-validate the cudamatrix suite on the RDNA part. Build with
`--rocm-targets=gfx1100` (no other source change expected). Do not re-plan from scratch.
