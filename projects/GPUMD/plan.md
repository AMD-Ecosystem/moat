# GPUMD ROCm/HIP port plan (linux-gfx90a lead)

Upstream: brucefan1983/GPUMD (master, HEAD c559721). GPU molecular dynamics:
NEP machine-learning potentials + classical (Tersoff etc.), builds `gpumd` and
`nep` executables. CUDA C++ (.cu/.cuh), no pytorch dependency.

## Existing AMD support assessment

GPUMD ALREADY ships an upstream HIP path. This is NOT a fresh CUDA->HIP port;
it is a "finish/validate/improve an existing port" job (PORTING_GUIDE "assess
existing AMD support", incomplete-or-below-best-practices bullet). Evidence:

- `src/makefile.hip`: `hipcc --offload-arch=gfx90a -DUSE_HIP`, links
  `-lhipblas -lhipsolver -lhipfft`. (the CUDA `src/makefile` uses nvcc + cublas/
  cusolver/cufft.)
- `src/utilities/gpu_macro.cuh`: the Strategy-A compat header. Aliases a
  `gpu*` symbol family to `hip*` under `USE_HIP`, else to `cuda*`/`cublas*`/
  `cusolver*`/`cufft*`. Covers memory, streams, error, device props, cuRAND,
  cuBLAS, cuSOLVER (Dn eigensolvers), cuFFT.
- 16 `USE_HIP` guard sites; the divergent ones select `thrust::hip::par` vs
  `thrust::cuda::par` (rocThrust drop-in) and a few HIP-specific headers.

So the upstream port is real and structurally sound. The MOAT contribution is:
(1) VALIDATE it actually builds and runs correctly on gfx90a under current ROCm
(upstream HIP makefiles routinely bitrot vs the ROCm in use), and (2) remove the
one clear best-practice gap: `makefile.hip` HARDCODES `--offload-arch=gfx90a`,
which forces every follower (gfx1100/gfx1151) to edit the file and churn the
head sha. Make the arch configurable with a `gfx90a` default.

Disposition: NOT `already-supported` for MOAT purposes (we still owe a real
gfx90a GPU validation + the arch-configurability fix); treat as a port to
validate and minimally improve.

## Build classification

Pure Makefile project (no CMake, no pytorch/Torch extension). Strategy A.
There is already a separate `makefile.hip`, so no language-gating CMake trick is
needed; the HIP build is just `make -f makefile.hip`. Strategy A's "isolate HIP
to the device TUs" goal is met trivially: every TU is a `.cu` and the whole
thing is compiled by hipcc (there is no separate host-C++ toolchain to protect).

## CUDA surface (what the port must cover)

- Libraries: cuBLAS (Sgemv/Sgemm/Sdgmm, DgemvBatched), cuSOLVER Dn dense
  eigensolvers (Dsyevj, Zheevj/Zheevd, batched), cuFFT (C2C 3D + Many), cuRAND
  (host+device normal RNG), Thrust (exclusive_scan, sort). All already swapped
  in the compat header to hipBLAS/hipSOLVER/hipFFT/hipRAND/rocThrust.
- Warp intrinsics: NONE. grep across all .cu/.cuh finds zero
  `__shfl*`/`__ballot`/`__any`/`__all`/`__activemask`/`__popc`/`warpSize`.
  Reductions are shared-memory block tree-reductions
  (`for (offset = blockDim.x>>1; offset>0; offset>>=1) { ...; __syncthreads(); }`,
  force.cu:292, neighbor.cu:674, nep_charge.cu:582) with `__syncthreads()` after
  EVERY step including sub-warp offsets -- no implicit warp-synchronous
  assumption. => wave64-safe by construction; the warp-size fault class does
  not apply.
- Textures/surfaces: NONE (zero `texture<`/`tex2D`/`surf2D`/`cudaArray`). The
  entire texture fault class (pitch, linear filter, layered coherency) is moot.
- Atomics: `atomicAdd` (21 files), `atomicCAS` (5 files). NO `atomicMin`/
  `atomicMax`. The gfx90a coarse-grained int atomicMin/Max-drop bug (cudaKDTree)
  does not apply; atomicAdd/atomicCAS are unaffected.

## Chosen strategy

Strategy A, minimal footprint. The port already exists and is correct in shape.
Planned changes (porter), all behind the existing HIP path, CUDA path byte-for-
byte unchanged:

1. `makefile.hip`: replace the hardcoded `--offload-arch=gfx90a` with a
   `HIP_ARCH ?= gfx90a` variable so followers build with
   `make -f makefile.hip HIP_ARCH=gfx1100` and no source/file edit churns the
   head sha. (PORTING_GUIDE "configurable HIP arch in the lead port".)
2. Only if the build surfaces a real ROCm 7.2 API/signature mismatch (e.g. a
   hipSOLVER/hipBLAS v2 enum/signature drift, or a missing compat alias) do we
   touch `gpu_macro.cuh`/wrappers, guarded by `USE_HIP`. Expect few or none.

No wave64 code changes anticipated (no warp intrinsics to fix). No texture
changes. No atomic emulation.

## Validation plan (real gfx90a GPU)

Build `gpumd` (and `nep`) with `make -f makefile.hip` on a free gfx90a, then:

- Run the shipped regression suite `tests/run_tests.sh`: each case runs a real
  MD trajectory and `diff -q`s the produced `thermo.out`/`dos.out`/`hac.out`/...
  against committed developer reference (`*1.out`). A clean diff is a strong
  bit-level correctness gate. Primary cases: `carbon` (NEP4 NVE, energy
  conservation), `graphene_dos` (Tersoff + DOS, exercises hipFFT/autocorr),
  `graphene_kappa_emd` (EMD thermal conductivity).
- Physical correctness from the `carbon` NVE run: columns are T, KE, PE; check
  total energy (KE+PE) conservation across the 10 dumped steps and temperature
  near the 300 K target.
- Determinism: GPUMD fixes the RNG seed from the input, so re-running the same
  case must reproduce thermo bit-for-bit; confirm run-to-run identical output.

Compile alone is explicitly NOT acceptance; the diff-vs-reference + energy
conservation + determinism on real gfx90a is the gate.

## Followers

gfx1100/gfx1151 reuse this branch and build with `make -f makefile.hip
HIP_ARCH=<arch>` once the arch is configurable; they are RDNA (wave32), but
since there are no warp-size assumptions in the code the lead fix should carry
over. They validate on their own hardware per MOAT policy.
