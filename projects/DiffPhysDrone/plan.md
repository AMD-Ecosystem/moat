# DiffPhysDrone -- ROCm/HIP port plan (lead platform linux-gfx90a)

## Project
- Name: DiffPhysDrone (Learning Vision-based Agile Flight via Differentiable Physics, Nature Machine Intelligence 2025)
- Upstream: https://github.com/HenryHuYu/DiffPhysDrone
- Default branch: master (HEAD 2719361 "update readme")
- The GPU code is a single PyTorch C++/CUDA extension `quadsim_cuda` under `src/`. It provides a differentiable drone flight simulator: a depth-render/raycast kernel, a nearest-obstacle-point collision kernel, and an analytic forward + analytic backward of the rigid-body flight dynamics (the "differentiable physics" core). Python training code (`env_cuda.py`, `main_cuda.py`, `model.py`) consumes the extension; only the extension is GPU code.

## Existing AMD support
- None. No HIP/ROCm/AMD reference anywhere in the tree (grep over .py/.cu/.cpp/.md is empty), single `master` branch, no stale port branch or fork. README documents only CUDA 11.8 + PyTorch 2.2.2.
- Decision: fresh CUDA-to-HIP port. It adds value (no prior HIP path, not even OpenCL/Vulkan). Disposition = PORT, mechanical (Strategy B). No skip.

## Build classification: torch-extension (Strategy B)
Evidence:
- `src/setup.py` lines 2,7: `from torch.utils.cpp_extension import BuildExtension, CUDAExtension` and `CUDAExtension('quadsim_cuda', ['quadsim.cpp','quadsim_kernel.cu','dynamics_kernel.cu'])` with `cmdclass={'build_ext': BuildExtension}`.
- No CMake anywhere; install is `pip install -e src` (README "Build CUDA Ops").
Building a `CUDAExtension` against a ROCm torch auto-runs `torch.utils.hipify` on the `.cu` sources and links amdhip64/c10_hip/torch_hip. So this is Strategy B: keep sources in CUDA spelling, let torch hipify them, fix only what hipify cannot.

## Port strategy: B (torch-hipify), mechanical, correctness-first
Rationale: the kernels are simple per-element arithmetic over small geometry arrays via `torch::PackedTensorAccessor`. There is NO performance-tuned NVIDIA-specific code (no CUTLASS/CuTe, no wgmma/MMA, no warp specialization), so no AMD-native (rocWMMA/CK/MFMA) rewrite is warranted. A mechanical hipify-driven build is the whole port. Expected source edits: zero to one (only if hipify leaves a `<cuda.h>` artifact; see risks). Do NOT add a compat header and do NOT hand-rename symbols (Strategy B rule).

## CUDA surface inventory
Six `__global__` kernels, all one-thread-per-output with a single `if (idx >= N) return;` guard, launched 1 block (dynamics/state) or grid-stride-less ceil-div blocks of 1024 (render/collision):
- `quadsim_kernel.cu`:
  - `render_cuda_kernel` -- per-pixel raycast min-distance against other drones, balls, two cylinder families, voxels (AABB slab test). Reads geometry arrays at `[batch_base][i][...]`.
  - `nearest_pt_cuda_kernel` -- per-(traj,drone) nearest obstacle point, same obstacle families.
  - `rerender_backward_cuda_kernel` -- per-pixel finite-difference of a depth buffer into a normalized gradient direction; reads a 2x2 neighborhood `depth[b][0][u*2..u*2+1][v*2..v*2+1]` (indices are in-bounds by construction: output H,W are half the depth H,W).
- `dynamics_kernel.cu`:
  - `update_state_vec_cuda_kernel` -- per-drone attitude/rotation-matrix update (normalize, cross product).
  - `run_forward_cuda_kernel` -- per-drone analytic forward dynamics step (drag, thrust, integrate p/v/a).
  - `run_backward_cuda_kernel` -- per-drone analytic backward (hand-written adjoint of the forward step).
- Host launchers use `AT_DISPATCH_FLOATING_TYPES` (float + double); the shipped test exercises the double path.
- Cross-lane / library surface: NONE.
  - No `__shfl*`, `__ballot`, `__any/__all`, `__activemask`, `warpSize`, cooperative groups.
  - No `__shared__`, no `__syncthreads`/`__syncwarp`.
  - No cub/hipcub, no Thrust (the only "thrust" tokens are a local physics scalar named `thrust` and a comment), no cuRAND/cuBLAS/cuFFT/cuSPARSE.
  - No textures/surfaces, no atomics, no managed/pinned memory, no explicit streams/events (kernels use the default/current stream).
  - No hardcoded `32`; grep for a bare `32` literal in the `.cu` is empty. The only block-size literal is `1024` (a legal block size on gfx90a).
- Math: device `sqrt/exp/pow/acos/abs/min/max` called unqualified (resolve to HIP device builtins on the device pass). `abs(scalar_t)` over `double` resolves to the floating overload.

## Risk list
This is a low-risk port -- none of the heavy fault classes apply -- but check:
1. WAVE SIZE (wave64 on gfx90a): NOT exposed. No warp/block collective, no `__shfl/__ballot`, no shared-memory reduction, no per-warp masks, no hardcoded 32. Each thread is fully independent. gfx90a wave64 vs CUDA warp32 cannot change results here. (This also means gfx1100/gfx1151 wave32 are equally safe -- see Delta plan note.)
2. `#include <cuda.h>` and `#include <cuda_runtime.h>` at the top of both `.cu` files. torch hipify maps `<cuda_runtime.h>`/`<cuda.h>` into the HIP runtime; the build resolves these via the ROCm toolchain. If hipify leaves an unresolved `<cuda.h>` (driver API header, sometimes not mapped), the single fix is to guard it `#if !defined(__HIP_PLATFORM_AMD__)` or drop it (these kernels use none of the CUDA Driver API -- only `<cuda_runtime.h>` symbols implied by torch). Low risk; first build will confirm.
3. `std::min`/`std::max` -> `::min`/`::max` hipify host-pass trap (gsplat lesson): NOT triggered. The kernels call bare `min`/`max` (not `std::min`), and these appear only inside `__global__` device code, never in a host TU. The host launchers (`run_forward_cuda` etc.) use no min/max. So the host-pass `::min(float,float)` failure mode does not arise. Confirm by a clean compile.
4. Float-literal / double-`scalar_t` mixing in `nearest_pt_cuda_kernel` (`max(1e-3f, oz+1)` with `oz` double; `min(-1., oz-1e-3f)`). Compiles on CUDA via overload resolution; same on HIP device builtins. Numeric result identical. No action unless a build warning escalates.
5. FP contraction / 1-ULP drift: the shipped test uses `torch.allclose` (rtol 1e-5, atol 1e-8) on DOUBLE tensors comparing the CUDA/HIP kernel against an independent PyTorch autograd reference -- this is a tolerance compare, not bit-exact, so the `-ffp-contract=fast` and `__fsqrt_rn` 1-ULP classes are comfortably absorbed. No `-ffp-contract=on` pin needed (Strategy B has no CMAKE_HIP_FLAGS hook anyway). If the double `allclose` ever flaps, that is the lead to investigate, not a presumed bug.
6. Missing-return / UB in the hand-written backward: `run_backward_cuda_kernel` is a hand-derived adjoint with many commented-out blocks. It is the highest-logic-density kernel. The shipped `test.py` is exactly the guard for this -- it checks all five returned gradients (`d_act_pred,d_act,d_p,d_v,d_a`) against `torch.autograd.backward` of the reference forward. A wrong adjoint or a UB read shows up there. (Note: this UB, if any, is platform-independent; it would fail on CUDA too. Our job is to confirm HIP matches CUDA's behavior via the same test.)
7. Out-of-bounds neighbor reads (colmap class): `rerender_backward_cuda_kernel` reads a 2x2 block of `depth` at `u*2,v*2`. By construction the output grid is half the depth resolution so the max index is `depth dim - 1`; not an edge-overscan stencil. No clamp needed. (The collision/render obstacle loops read `[batch_base][i]` strictly within `.size(1)`.) No action; note for the reviewer.
8. `rerender_backward` is invoked from Python only in some code paths; the GPU gate (test.py) covers run_forward/run_backward directly. render/find_nearest_pt are covered by an environment-level smoke run (see test plan).

## File-by-file change list
Expected: NO source edits in the common case (Strategy B; hipify does the translation at build time). Contingent, only if the first ROCm build fails:
- `src/quadsim_kernel.cu`, `src/dynamics_kernel.cu`: only if hipify leaves an unmapped `<cuda.h>`, guard or remove that one include (risk 2). No other edits anticipated.
- `src/setup.py`: no change expected. If a flag is needed (unlikely), prefer leaving CUDAExtension defaults so the NVIDIA path stays byte-identical.
The porter must NOT add a cuda_to_hip.h compat header (that is Strategy A) and must keep all sources in CUDA spelling.

## Build commands (gfx90a)
ROCm torch already present on the host (torch 2.13.0a0, HIP 7.2.53211, torch.cuda.is_available()=True).
```
# from a clean checkout of the fork's moat-port branch
export PYTORCH_ROCM_ARCH=gfx90a
cd projects/DiffPhysDrone/src/src     # the extension lives in src/ which contains setup.py
pip install -e .                       # CUDAExtension -> auto-hipify -> build quadsim_cuda
# Incremental-build gotcha (PORTING_GUIDE Strategy B): if a .cu is edited after a first
# hipify, remove the stale hipified mirror / build dir before rebuilding:
#   rm -rf build *.egg-info && pip install -e . --no-build-isolation
```
A CPU-only docker compile check (image rocm/dev-ubuntu-24.04:7.2.4-complete) is optional and is NOT a gate.

## Test plan (real GPU gate)
PRIMARY GPU GATE -- the repo's own `src/test.py`:
- It builds 64 random DOUBLE-precision drone states on `cuda`, runs `quadsim_cuda.run_forward` and compares all four outputs (`act_next,p_next,v_next,a_next`) to an independent pure-PyTorch reimplementation via `torch.allclose`; then seeds random output-gradients, runs `torch.autograd.backward` on the reference to get gold grads, runs `quadsim_cuda.run_backward`, and asserts `torch.allclose` on all five gradients. This is precisely the "forward sim + analytic-backward vs autograd" correctness check and is the validation gate.
  - Run: `cd projects/DiffPhysDrone/src/src && python test.py` -- must exit 0 (all six asserts pass) on real gfx90a.
  - Determinism: kernels are atomic-free and per-thread independent, so two runs with a fixed seed must be bit-identical; assert that as a secondary check (set `torch.manual_seed(0)` in a small harness wrapping test.py's tensors).
- SECONDARY GPU coverage (render + collision kernels, not in test.py): drive `env_cuda.py`'s `QuadSim`/environment a few steps on `device='cuda'` so `quadsim_cuda.render`, `find_nearest_pt`, `update_state_vec`, and `rerender_backward` all execute without fault and produce finite tensors; compare a forward render against a small CPU/torch reference of the slab/sphere raycast for a handful of pixels, or at minimum assert finite + run-to-run identical. (A full training run via `main_cuda.py` with `configs/*.args` is the integration-level exercise; one short iteration suffices to prove the kernels run end-to-end under autograd.)
- NON-GPU regression set: there are no CPU-only unit tests in the repo; the only test is GPU. Nothing to regress. Just ensure `import quadsim_cuda` succeeds and the Python driver imports.

## Disposition
PORT (Strategy B, mechanical). Advance linux-gfx90a to `planned`. Lead bringup designed to be wave-agnostic so gfx1100/gfx1151 should validate with no code change.

## Delta plan note for followers (gfx1100 / gfx1151, RDNA wave32)
No delta expected. The port touches zero wave-width-sensitive constructs (no shfl/ballot/shared-mem reduction/warp masks/hardcoded 32), so the same `moat-port` branch should build and pass `test.py` on wave32 with only `PYTORCH_ROCM_ARCH=<arch>` changed at build time. If a follower fails, it would be a torch-on-RDNA build/runtime issue, not a port logic issue; do not re-plan -- append findings here.
