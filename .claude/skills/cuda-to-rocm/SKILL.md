---
name: cuda-to-rocm
description: Port a CUDA project to ROCm/HIP for AMD GPUs. Strategy selection, the AMD-strict-where-CUDA-is-lenient fault classes, wavefront-size portability, library substitutions, and the validation bar. Use when planning, implementing, reviewing or validating a CUDA-to-HIP port, or when a HIP build compiles but misbehaves at runtime.
---

# CUDA to ROCm/HIP

Accumulated across every port MOAT has done. This file is the index and the decision-critical parts;
everything else is in `references/`, loaded only when you need it.

**Most porting bugs are not in symbol names.** Hipify handles the renaming. The bugs are
semantic differences where AMD is strict and CUDA is lenient, and they surface at runtime
on a build that compiled cleanly. Read the fault-class index below before concluding a
port works.

## Naming: ROCm or HIP?

They name different things and reviewers notice.

- **HIP** is the programming model: the kernel dialect plus the `hipXxx` runtime API. The
  analogue of CUDA C++ and the CUDA runtime.
- **ROCm** is the platform: compiler, runtime, driver, and the roc*/hip* libraries. The
  analogue of the CUDA Toolkit.

So the CODE port is "to HIP" (hipify, the compat header, runtime symbols), while the
TARGET, build flag and libraries are "ROCm" (`USE_HIP`/`USE_ROCM`, "ROCm 7.2.1",
cuFFT -> hipFFT). A language+runtime port is precisely "a HIP port targeting ROCm". Never
call the platform "HIP" or the kernel dialect "ROCm". Details: `references/naming.md`.

## Which strategy

Classify the build first -- implementing the wrong strategy correctly is still wrong.

| build | strategy | shape |
|---|---|---|
| pure CMake | **A** (preferred) | one `cuda_to_hip.h` compat header, `enable_language(HIP)`, `set_source_files_properties(... LANGUAGE HIP)`; sources keep CUDA spelling |
| pytorch extension | **B** | rely on torch's build-time hipify; fix only what it cannot |
| anything else | neither | driver-API, runtime PTX, Go/cgo, qmake and codegen builds exist -- see the runtime-PTX fault class |

**How to tell:** look for `find_package(Torch)`, `torch.utils.cpp_extension`,
`CUDAExtension`, or a torch dependency in `setup.py`/`pyproject.toml`. If any is present it
is a pytorch extension -- Strategy B. Otherwise treat it as a pure CMake (or Makefile, or
meson) project -- Strategy A. The build system does not pick the strategy; what the sources
look like does. lc0 is meson and ports Strategy-A-shaped -- it just means writing by hand
what `enable_language(HIP)` would have given you, and `references/strategy-a-cmake.md` says
what.

Strategy A is the minimal-footprint model: the compat header is a no-op on NVIDIA, so the
CUDA build is untouched. Where the project includes CUDA headers by name, prefer the
**shim-header method** -- shadow headers on the HIP include path only -- which leaves the
source tree byte-identical to upstream and produces a far more mergeable diff. Full recipes: `references/strategy-a-cmake.md`,
`references/strategy-b-torch.md`.

## Before porting

Check whether an AMD port already exists, and whether it is authoritative. The deciding
axis is authoritativeness, not existence -- an AMD-official work-in-progress shifts the
value to validating and improving it, while a one-off community fork is a hint, never code
to inherit. Cheapest high-signal check first: grep the upstream's own docs for
`amd|rocm|hip|gfx[0-9]`, because reference repos routinely link platform forks.
`references/assess-existing-support.md`.

Some CUDA has no HIP analogue to translate INTO, so hipify succeeds at doing nothing and the port looks small. CUTLASS reimplements against Composable Kernel; CuTe has a real structural correspondence in AMD's FlyDSL, whose layout algebra is modelled on it. That is a reimplementation with a defined target, not the same verdict as "no target exists" -- and both get written `cant-port` unless you say which. `references/no-hip-equivalent.md`.

## Fault classes -- the index

Scan this list against what you are doing. If any line could apply, open
`references/fault-classes.md` for the full entry with evidence and fix.

**Wavefront and warp semantics**
- Warp size: NVIDIA is always 32; AMD is 64 on CDNA (gfx9xx) and 32 on RDNA. Never hardcode.
- Warp-derived array sizing: quantities scaling WITH `warpSize` need the 64 upper bound, quantities scaling as `blockDim/warpSize` need the 32 LOWER bound. Getting it backwards writes out of bounds.
- A compat `__ballot_sync` that casts `__ballot()` to `uint32_t` returns the wrong 32 lanes on wave64.
- An over-wide `__shfl*` width silently clamps to the physical wavefront instead of erroring; the symptom is a metric shift, not a crash.
- Intra-wave barrier divergence: a per-row early return before `__syncthreads()` is benign on CUDA, faults on wave64.
- `cub`/`hipCUB` block-collective `TempStorage` reuse races on a 64-thread block without an explicit `__syncthreads`.
- `__smid()` can EXCEED `multiProcessorCount` on AMD, unlike NVIDIA.
- HIP's device `warpSize` is not an `int`; cast it at any `printf` vararg.

**Memory and lifetime**
- Out-of-bounds reads: CUDA often tolerates one element past an allocation; AMD faults.
- Fresh device allocations are NOT zeroed on ROCm, where CUDA's allocator often hands back zeroed pages.
- `cudaMemcpy*Async` from a soon-freed pageable host buffer: CUDA stages synchronously, HIP does not.
- Rule-of-five on resource handles: CUDA tolerates double-destroy of a texture/stream/event; AMD does not.
- A functor returning a reference to a by-value parameter (lifetime UB) is tolerated by nvcc, not by clang/HIP.
- `hipLaunchHostFunc` callback threads must not call ANY runtime API.

**Textures**
- Texture pitch alignment: AMD requires 256-byte row pitch for pitched 2D binds (32 on NVIDIA).
- Layered `cudaArray` collapses across kernel launches -- use a non-layered 3D array (confirmed gfx90a bug).
- Hardware linear-filter texture over an element-read float array is arch-specific, not a blanket AMD limitation.
- Gate hardware-vs-software texture paths on a VERIFIED runtime self-test, never on creation success or `_WIN32`.

**Floating point**
- `__fsqrt_rn` is not always correctly rounded on gfx90a (1 ULP high); CUDA's is IEEE-correct.
- clang(HIP) defaults to `-ffp-contract=fast` and forms FMAs ACROSS expressions; nvcc contracts expression-only.
- Exact float-equality branches fed by `__fdividef` produce out-of-range indices.
- HIP device `cuda::min`/`max` NaN-selection can differ from host `std::min`/`max`.

**Headers, includes and build**
- A shared compat header must be host-includable: gate device-only includes behind `__CUDACC__`/`__HIPCC__`, or CUB leaks into host TUs. Hit independently by two projects.
- `__HIP_PLATFORM_AMD__` is undefined until `hip_runtime.h` is included; a gate placed before it silently picks the CUDA branch.
- Missing includes in a HIP port are usually pre-existing upstream omissions unmasked by the narrower include graph.
- A `-include`d compat header creates no dependency edge: wipe objects after editing it or you validate stale code.

**Types, dispatch and platform limits**
- Never name the pointee struct of an opaque handle (`CUstream_st`); use `std::remove_pointer_t<cudaStream_t>`.
- CUPTI has no ROCm analogue; ck_tile fmha ships headers but no instance library.
- `char` vs `signed char` vector base types differ between HIP and CUDA.
- AMD compute-capability values COLLIDE with NVIDIA arch numbers, so CC-keyed dispatch picks the wrong path.
- Library swaps: cuBLAS -> hipBLAS, cuFFT -> hipFFT, cuRAND -> hipRAND, cuSPARSE -> hipSPARSE, cuDNN -> MIOpen, Thrust/CUB -> rocThrust/hipCUB. Handle types and v2-enum signatures differ.
- Runtime PTX + CUDA Driver API is a third build class beyond A and B.
- MSVC-only upstreams accept code gcc/clang reject; the HIP build is a stricter compiler.
- No graphics pipeline on compute-only CDNA: OpenGL/Vulkan-interop apps build but cannot run on gfx90a.
- An arch-specific fix keyed on OS or ROCm version can BREAK an already-validated arch.

Two diagnostic methods in `references/validation.md` are worth knowing before you escalate a
suspected AMD fault: a per-tile "later data corrupts earlier" signature usually means the
reproducer's input dtype is wrong, and SP-vs-DP triangulation distinguishes FP reassociation
from a genuine wavefront bug. Both were learned the expensive way.

## Portability rule for shared code

Any fix to code that is not arch-guarded must be correct on **both** wavefront sizes. A
per-arch hack that fixes wave32 and regresses wave64 makes the archs ping-pong, and each
"fix" then costs a full revalidation everywhere. Prefer a runtime `warpSize` query on the
host and a compile-time constant in device code.

## Validation bar

A real-GPU test run is the only evidence that counts. Lint is not validation, and a
CPU-only build proves compilation and nothing else. Coverage is expressed as gates --
wave64, wave32, windows -- satisfied by any arch carrying that attribute. Also compile the
CUDA path with nvcc as a no-regression gate: an additive ROCm port must leave the CUDA
build a pure passthrough. `references/validation.md`.

## Writing it up

Commit messages, PR bodies and code comments go to upstream maintainers who do not know
our vocabulary. Check with `python3 utils/jargon.py --port <name>` before pushing -- the whole branch,
not the commit you just wrote; terms and their replacements are in `config/jargon.toml`. A Test
Plan runs from a clean clone of the upstream project, so it carries no control-plane paths --
jargon.py skips fenced blocks by design and cannot see them. `references/naming.md`.
