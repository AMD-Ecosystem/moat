# Fault classes: AMD is strict where CUDA is lenient

The real semantic differences. Most porting bugs are here, not in symbol names, and they
surface at runtime on a build that compiled cleanly.

Sections match the index in SKILL.md, so a line that looked relevant there leads straight
to its entry here. Each entry names the project it was learned on; the full incident lives
in that project's `notes.md`.

## Wavefront and warp semantics

**Never hardcode 32.** NVIDIA warps are always 32 lanes. AMD wavefronts are 64 on CDNA
(gfx90a, gfx94x) and 32 on RDNA (gfx10xx, gfx11xx, gfx12xx).

- Host code (launch geometry, host-side shared-memory sizing) queries at runtime:
  `hipGetDeviceProperties(&prop, dev); prop.warpSize`, or `hipDeviceAttributeWarpSize`.
  PyTorch exposes `at::cuda::warp_size()`.
- Device code uses a compile-time per-arch constant. `__GFX9__` is defined only during
  device compilation; there is no `__AMDGCN_WAVEFRONT_SIZE__` in ROCm 7.2.x, so the
  `__GFX*__` guards are the supported selector:

        #if defined(__HIP_PLATFORM_AMD__)
        #if defined(__GFX9__)
        static constexpr int kWarpSize = 64;   // CDNA: gfx90a, gfx94x
        #else
        static constexpr int kWarpSize = 32;   // RDNA: gfx10xx, gfx11xx
        #endif
        #else
        static constexpr int kWarpSize = 32;   // CUDA
        #endif

- **Never route both host and device through one shared constant.** A CMake-injected
  `-DWARP_SIZE=64` forces single-arch and mis-sizes host buffers in a fat binary. Device
  width comes from the `__GFX*__` guards, host width from the runtime query.

**A static array sized from the warp width needs the bound that matches the DIRECTION the
width enters the formula, and getting it backwards writes out of bounds.** A quantity
scaling WITH the width (`n * warpSize`, a per-lane array) needs the UPPER bound, 64
(PyTorch's `C10_WARP_SIZE_UPPER_BOUND`). A quantity scaling INVERSELY (`blockDim /
warpSize`, warps-per-block, anything sized per-warp) needs the LOWER bound, 32: a smaller
wavefront means MORE warps, so a 64 bound UNDER-allocates on wave32 and the device writes
past the shared-memory region. Ask which way the divide goes before picking the constant.
(SCAMP: `cov_handoff`, `2 * warps_per_block` scalars sized with 64, under-allocated on
every wave32 device.)

**Lane masks must be 64-bit where the API takes one** -- `__shfl*`, `__ballot`,
`__activemask`. A `uint32` mask is wrong on a 64-wide wavefront.

**And the mask VALUE must be built in 64 bits, which the destination type does not do for
you.** In `const uint64_t mask = ((1 << K) - 1) << (K * subgroup);` the shift result type is
the promoted LEFT operand, i.e. `int`, so the whole expression is evaluated in 32 bits and
the widening to `uint64_t` happens afterwards, too late. Shifting an `int` by 32 or more is
undefined, and AMD does not trap or saturate: the 32-bit shift instruction uses only the low
5 bits of the count, so the count wraps and lanes 32..63 silently get the mask belonging to
lanes 0..31. Any mask literal that can be shifted past bit 31 needs `1ull`. The class is
nasty because it compiles clean, is invisible on wave32 (a 32-lane wavefront never reaches a
shift of 32, which is why upstream RDNA testing does not catch it), and yields a wrong-lane
collective rather than a fault. Grep every `1 <<` and `1u <<` whose shift count is derived
from a lane index. HIP RT 3.1.0.cb09c56 has exactly this in three places
(`hiprt/impl/BvhBuilderKernels.h`, `subwarpMask`): the upper half of the wavefront selects
its widest child from the wrong subgroup, one subtree gets opened twice, the collapse
therefore emits more leaf references than there are primitives, and the persistent builder
kernel -- whose only exit is `emittedCount == primitiveCount` -- steps over its own exit
value and spins forever, hanging the GPU inside a BVH build over a few dozen triangles.
(diff-surfel-tracing, in its HIP RT dependency.)

**Do not hardcode wave64 GEOMETRY either** -- the inverse failure. Packing two logical
32-thread rows into one wavefront (`wflane = threadIdx.x + ((threadIdx.y&1)<<5)`, a
`1ull<<wflane` prefix mask, `__shfl(...,0,64)`, `__popcll` over a 64-bit ballot, a
`wflane==0` leader test) COMPILES on wave32 and silently miscomputes: each 32-thread row is
its own wavefront there, so odd-row lanes land at 32..63, the leader test never fires
(lost atomics), and the mask reads phantom bits. Make wave-collective geometry
`warpSize`-generic, or give it an explicit `__GFX9__` branch AND a wave32 `#else`. A
wave64-only fix that passes CDNA is not done until it is correct on RDNA. (popsift
s_extrema.)

Width-32 LOGICAL warp ops (`__shfl*(...,32)`, `cub::WarpReduce<,32>`,
`cg::tiled_partition<32>`) are arch-agnostic and fine -- they work within a 32-lane
subgroup whatever the physical width.

**A compat `__ballot_sync` that casts `__ballot()` to `uint32_t` takes the LOW 32 lanes**,
which are the wrong lanes for any logical warp not at the wavefront base -- with
`blockDim (32,32)`, odd `threadIdx.y` rows sit in the high lanes. Arch-unified form:
`(uint32_t)(__ballot(PRED) >> (__lane_id() & ~31u))`, where the shift is 0 on wave32. (cuSZ)

**An over-wide `__shfl*` width does not error, it silently degenerates.**
`__shfl_up(v, delta, 64)` lowers to `__builtin_amdgcn_ds_bpermute` with a clamp of
`self & ~(width-1)`; on wave32 that clamp is 0, so the permute domain becomes the
32-lane physical wavefront while the loop still treats lanes as a 64-wide group. In a
lossy or statistical pipeline the wrong collective still round-trips, so it surfaces as a
quality-metric shift (compression ratio, PSNR) rather than a crash -- the cross-arch gate
has to compare the METRIC, not just that the output decodes. (cuSZ)

**Intra-wave barrier divergence.** A per-row early `return` before `__syncthreads()` is
benign on CUDA when a row is a whole 32-lane warp -- the warp exits and leaves the barrier
wait set. On wave64 two rows share a wavefront, so one row returning is a half-wave
divergence and the wave stays live with lanes missing. Audit every block barrier whose
early-return granularity is under 64, and fold the return into an out-of-bounds predicate
so all lanes reach the barrier. (lc0)

Detector for it: route `__syncthreads()` through a wrapper that, in HIP debug builds only,
computes the expected active-lane mask from the block layout and asserts
`__ballot(1) == expected` first. It does not false-positive on a legitimately partial final
wave. If device code is compiled by explicit hipcc command lines, mirror the release
`-DNDEBUG` yourself or the asserts stay on in release. (lc0)

**`cub`/`hipCUB` block collectives race on a 64-thread block without an explicit
`__syncthreads()`.** At wave64 a 64-thread block is a single wavefront, so a `BlockReduce`
or `BlockRadixSort` lowers to a single-wavefront op with no syncing epilogue; reusing one
`TempStorage` union across back-to-back calls lets the second call's writes overlap the
first's reads. CUDA's 32-wide warp split the block in two and masked it. (CV-CUDA:
OpPairwiseMatcher crossCheck.)

**`__smid()` can EXCEED `multiProcessorCount` on AMD.** NVIDIA's `%smid` is dense in
`[0, multiProcessorCount)`, so host code routinely sizes a per-SM scratch pool to that
count and indexes it `pool + smid()*POOL_SIZE`. AMD packs `(SE,CU)` sparsely -- gfx9 is
`(se_id << 4) | cu_id`, so values reach 127 with ~104 CUs active (measured: max 125 on a
gfx90a GCD with `multiProcessorCount` 104). A pool sized to the CU count is written past
its end by any block on a high-id CU: a SILENT out-of-bounds device-heap write that
corrupts a later read, so a green test run does not prove safety. Size such pools by a
one-time runtime probe -- a saturating kernel that `atomicMax`es `get_smid()`, take
`max+1`, round up, cache it -- rather than hardcoding bit layouts. Do NOT `% workers` the
index: that reintroduces the contention the private-slot pool exists to prevent. Audit
every allocation paired with a `smid()*STRIDE` kernel; pools indexed by `blockIdx.x` with a
`gridDim == multiProcessorCount` launch are self-consistent and fine. Keep the CUDA path at
`multiProcessorCount`. Prove the fix with a sentinel region after the pool. (gpu4pyscf:
int3c2e fill pool, ~4MB silent OOB.)

**HIP's device `warpSize` is not a plain `int`** and device `printf` rejects it as a
vararg: cast it, `printf("%d", (int)warpSize)`. The raw diagnostic is misleading -- it
surfaced as an undeclared-identifier cascade in a generated dispatch header. No-op on
CUDA. (LC-framework)

**Validate wave-size work on BOTH widths, and where there is no host reference use a
CROSS-ARCH CONSISTENCY gate.** "Deterministic, non-zero and plausible" passes a wrong
wave32 result -- popsift's gfx1100 reported deterministic non-zero features from a
miscounted extrema path, a false pass. Where the algorithm is deterministic, diff the
wave32 output against the wave64 output for the same input. (popsift)

**Special case: a warp-width-dependent SERIALIZED FORMAT.** If the warp width determines an
on-disk or in-memory layout rather than just launch geometry, it cannot follow the per-arch
device constant -- pin it to a fixed maximum width, or carry the producing device's width
at runtime. dietgpu's rANS archive serializes one coder state per lane
(`ANSWarpState::warpState[kWarpSize]`), making its header geometry width-coupled; pinning
`warpState[64]` keeps every `sizeof()` and header offset arch-independent, leaves the
wave64 archive byte-identical to its old format, and lets a wave32 device use the first 32
slots. (dietgpu)

## Memory and lifetime

**Out-of-bounds reads.** CUDA often tolerates a read one element past an allocation; AMD
faults. Kernels reading index +/-1 or +/-width at edges (stencils, neighbour gathers) must
clamp. (colmap ComputeDOG.)

**Rule-of-five on resource handles.** CUDA tolerates a default-constructed or
double-destroyed texture/stream/event handle; AMD faults. Give RAII wrappers explicit
default init (`handle = 0`), move-only semantics, and a guarded destructor. (colmap
CuTexObj.)

**Fresh device allocations are NOT zero on ROCm**, where CUDA's allocator often hands back
zeroed pages. A kernel writing only a sub-region and implicitly relying on a zero start
passes on CUDA and reads garbage on ROCm. Fingerprint: a zero-tolerance test that PASSES in
isolation and FAILS in-suite, after earlier tests have dirtied and freed similar buffers.
The reliance is a latent upstream bug, not a ROCm defect: zero explicitly
(`hipMemsetAsync`) before the partial-write kernel, and audit any kernel whose write set is
smaller than its output allocation. (opencv_contrib: cudastereo census_transform border.)

**`cudaMemcpy*Async` from a soon-freed pageable host buffer.** CUDA stages pageable async
copies synchronously, so a test that copies from a local `std::vector` and lets it go out of
scope still works. ROCm's is genuinely asynchronous, so the freed and reused buffer is read
after the copy is enqueued. The corruption appears only in bytes the kernel does not
overwrite, and the failing set varies run to run. This is a use-after-free that is latent UB
on CUDA, not an allocator defect -- confirm by keeping the source alive and watching the
failures vanish. (CV-CUDA: InterpolationVarShapeWrap.)

**A functor returning a reference to a by-value or forwarded parameter** binds to a
temporary that expires at the return. nvcc and CUB happen to keep the value live in
registers; rocPRIM block-loads the dangling reference and reads garbage. Return BY VALUE --
it is UB on CUDA too. Localise it by computing the same reduction three ways (library
wrapper, raw hipCUB call, `thrust::reduce` over the same iterator plus the bare operator):
if only the wrapper is wrong, the bug is in its transform, not the iterator or rocPRIM. A
standalone microbench often will NOT reproduce it, because the dangle manifests only
through the library's nested iterator instantiation (here a `thrust::transform_iterator`
feeding `hipCUB DeviceReduce::Reduce`). (cudf: `cast_functor.cuh` same-type
`operator()(T&&)`, selected only when reduction output type equals input element type --
min/max returned the seeded identity, float sum returned nan, int32->int64 sum was correct.)

**`cudaLaunchHostFunc`/`hipLaunchHostFunc` callback threads must not call ANY runtime API** -- an RAII deleter
that frees inside one DEADLOCKS ROCm. HIP forbids it as CUDA does, but CUDA more often
tolerates it, so only AMD hangs. The trap is indirect: a smart pointer whose deleter IS a
runtime call (`hipFree`, `hipStreamDestroy`, `hipEventDestroy`, `hipHostUnregister`) drops its last reference
inside the callback. The symptom masquerades as a slow kernel -- the GPU pegs at 100% while
a runtime worker spins, and serialising the API (`AMD_LOG_LEVEL=3`) slips past the deadlock
window, so the same test alternates between hanging and returning a fast raced result. Do
not dismiss it as a slow test; an op orders of magnitude slower than the CPU engine is a bug
to root-cause, never one to exclude. Detect it with a gdb backtrace of the live hang: the
HIP worker sits in your deleter -> `hipFree` -> `sched_yield` inside `libamdhip64`. Fix by
keeping the callback API-free -- move the handles into a deferred-free list (a container
move, no API call) and drain it from a user thread that already issues runtime calls, under
the existing queue mutex, draining on teardown too. Only the op that hands per-call
allocations into a callback-retired queue trips this, which is why one gate hangs while
everything else passes. (qrack: `_PopQueue` under `UniformlyControlledSingleBit`.)

**`hipFree` is synchronizing** (`hipFreeAsync` is not), so an explicit
`hipDeviceSynchronize()` before it is redundant. (anari-visionaray)

## Textures

**Texture pitch alignment is 256 bytes on AMD against 32 on NVIDIA**, and it bites in two
distinct ways.

- At the BIND: pitched 2D texture binds need 256-byte rows, so widths that work on CUDA can
  fail. If a kernel only point-samples, a linear (`tex1Dfetch`-style) bind avoids pitch
  entirely. (colmap BindTexture2D.)
- Through the ATTRIBUTE: `cudaDevAttrTexturePitchAlignment` (`hipDeviceAttributeTexturePitchAlignment`) reports 256 on gfx90a,
  so libraries deriving a row pitch from it pad more on AMD -- a tight 640-byte uchar row
  becomes 768. Tests that fill the valid region, run the op, then compare the WHOLE strided
  buffer against a zeroed reference bake in the CUDA pitch and mismatch. If nothing is
  actually bound to a hardware texture object the 256B pitch is unnecessary: clamp the
  attribute query to 32 in a compat shim so AMD pitches match CUDA. (CV-CUDA: Erase, SIFT,
  Gaussian, FindHomography, Normalize, TensorBatchWrap, many Interpolation cases.)

**A layered `cudaArray` collapses across kernel launches -- use a non-layered 3D array.**
CONFIRMED bug, gfx90a/CDNA2, ROCm 7.2.1, ROCm/clr#275. A
`cudaArrayLayered | cudaArraySurfaceLoadStore` float array written one layer at a time via
`surf2DLayeredwrite` reads back the LAST-written layer for EVERY layer index on a later
launch -- through `tex2DLayered`, `surf2DLayeredread` and host `hipMemcpy3D` alike.
`hipDeviceSynchronize` between writes and recreating the texture do not help. Fix: drop
`cudaArrayLayered` and allocate with `hipMalloc3DArray`, accessed via
`surf3Dwrite`/`surf3Dread`/`tex3D` with the layer as a real z coordinate (a tall 2D array,
W x H*L, also works). Map each `surf2DLayeredwrite(v,s,x,y,layer)` to
`surf3Dwrite(v,s,x,y,layer)` once inside the compat header so call sites are untouched.
CUDA keeps the real layered array byte-for-byte. AMD's candidate fix
(ROCm/rocm-systems#6683) corrects only the `surf2DLayered` builtins -- `tex2DLayered` uses a
different builtin and may collapse independently, so re-verify per ROCm version and keep
the workaround until proven fixed on your stack. NOT confirmed on RDNA; a wave32 porter
should re-run the repro to establish arch scope. (popsift: the Gaussian pyramid and DoG were
layered arrays, and without this the DoG was all-zero.)

**A hardware linear-filter texture over an element-read float array is ARCH-SPECIFIC, not a
blanket AMD limitation.** On gfx90a (ROCm 7.2.1), `filterMode=cudaFilterModeLinear` with
`readMode=cudaReadModeElementType` over a float array is REJECTED at creation
(`hipCreateTextureObject` -> "operation not supported"). On gfx1100 (ROCm 7.2.1) and
gfx1201 (ROCm 7.14) the same texture is ACCEPTED and genuinely hardware-interpolates, so
RDNA needs no fallback. For the rejecting arch, create it `cudaFilterModePoint` and lerp in
software, matching CUDA's unnormalized -0.5 texel-center convention: coordinate c samples
`floor(c-0.5)` and `floor(c-0.5)+1`, weight `(c-0.5)-floor(c-0.5)`. Put it behind the
project's fetch helper, `__GFX9__`-gated. Empirical, no upstream issue filed. SCOPE: this is
a property of the hardware TEXTURE OBJECT, not of linear filtering as such -- a field
sampled through a software accessor in device code (a NanoVDB `getAccessor().getValue()`)
is unaffected and needs no fallback. (popsift readTex; TIGRE probe; anari-visionaray.)

**Gate the hardware-vs-software texture path on a VERIFIED runtime self-test**, never on
creation success or an `_WIN32`/version proxy -- and probe the EXACT configuration you use.
Detecting at runtime lets one binary use hardware where present, which is real perf: the
interpolated fetch can be ~90% of kernel time. Two traps. Creation success is NOT proof:
some hardware accepts a linear fp32 texture and silently point-samples, so the self-test
must SAMPLE a known ramp and confirm it interpolates (1.5 at the midpoint, not 2.0). And
the probe must mirror the real configuration exactly -- on gfx90a/ROCm 7.2.1 a small 1D
`cudaAddressModeClamp` linear texture is ACCEPTED while the real 3D,
`cudaAddressModeBorder`, `cudaReadModeElementType` one is REJECTED, so a 1D probe gives a false "supported" and the real texture
then fails at creation. Sample an interior texel so border padding does not pollute the
lerp. Drive both the `filterMode` and the fetch helper from one cached verdict so they
cannot disagree. RDC-free mechanism: a header-local `static __constant__` flag plus a
`static inline` host sync helper using `hipMemcpyToSymbol(HIP_SYMBOL(flag), ...)`, so each
TU binds its own copy. The once-per-process init log also makes the port self-report each
arch's verdict during validation. (TIGRE `cuda_to_hip.h` + gpuUtils.cu.)

## Floating point

**`__fsqrt_rn` is not always correctly rounded on gfx90a** (1 ULP high) where CUDA's
`sqrt.rn.f32` is IEEE correctly-rounded, so bit-exact tests pass on NVIDIA and fail on
gfx90a. Route the f32 device sqrt through the correctly-rounded f64 `__dsqrt_rn` and cast
back; CDNA has fast f64 sqrt so the cost is negligible. (CV-CUDA: OpNormalize, L2
PairwiseMatcher.)

**clang for HIP defaults to `-ffp-contract=fast`** and forms FMAs ACROSS statements, where
nvcc with `--fmad=true` contracts only within one expression. The extra contraction drifts
results ~1 ULP from the CUDA build and CPU gold. MATCH UPSTREAM's setting rather than
pinning one unconditionally: CV-CUDA needed `-ffp-contract=on` to match nvcc, while
LC-framework's nvcc recipe uses `-fmad=false -mno-fma -ffp-contract=off`, so pinning `on`
there would diverge. Integer-only components are unaffected. (CV-CUDA OpWarpPerspective;
LC-framework.)

**An exact float-equality branch fed by approximate division can HANG, not just drift.**
HIP's `__fdividef` differs ~1 ULP from CUDA, so a downstream exact-equality test on the
quotient -- selecting a loop start or end index -- can flip on AMD. When that index feeds an
UNSIGNED loop bound via subtraction, the out-of-range value underflows to a huge count and
the loop never terminates: a data-dependent hang, not a wrong-by-1-ULP number. Do not gate
control flow on exact equality of approximate fp; use a tolerance, or cap the derived bound
at its provable geometric maximum. This is the class where numeric drift becomes a
CONTROL-FLOW failure. (TIGRE Siddon_projection: `Np` derived from `__fdividef` comparisons,
a 0.2s kernel ran >10 min at some angles; capped at `Nx+Ny+Nz+3`, a no-op for valid rays and
identical on CUDA.)

**Device `cuda::min`/`cuda::max` NaN-selection can differ from host `std::min`/`std::max`.**
Host `std::min(a,b)` is `b<a?b:a` and `std::max(a,b)` is `a<b?b:a`; a device version written
`a<b?a:b` picks the OPPOSITE operand on a NaN compare, so tests filling buffers with random
bytes and comparing host gold against device output differ bit-for-bit. Spell the device
ternaries to match the host forms exactly. (CV-CUDA: OpMorphology CLOSE on RGBAf32.)

## Headers, includes and build

**A shared compat header must be host-includable.** Host `.cpp` TUs reach the shim through
ordinary headers, so an unconditional `#include <cub/...>` or `<hipcub/...>` leaks device
headers into g++ and fails there. Put the shim in the lowest common layer, keep host-safe
includes (`hip_runtime.h`, `hipfft.h`) unconditional, and gate device-only ones behind
`__CUDACC__` or `__HIPCC__ || __HIP_DEVICE_COMPILE__`. (SCAMP, stdgpu)

**`__HIP_PLATFORM_AMD__` is undefined until `hip/hip_runtime.h` has been included in that
TU.** A wave-width gate in a header included BEFORE the runtime header silently takes the
CUDA branch and picks width 32. hipify-perl prepends the runtime include at line 1, which
masks the bug -- so the gate is correct only by accident and breaks for anyone building
without hipify. Verify include order rather than trusting the prepend. (LC-framework)

**Missing-include errors are usually pre-existing upstream omissions**, not port breakage:
CUDA's headers supplied them transitively and the narrower HIP include graph unmasks them.
Fix additively -- it helps the CUDA build too. (LC-framework)

**A force-included compat header creates no build dependency edge.** After editing a header
injected with `-include`, object files are NOT rebuilt: wipe them manually or you validate
stale code and get a silent false pass. (lc0)

**MSVC-only upstreams accept code that clang and gcc reject**, so the HIP build (and the
CUDA build under nvcc) is a stricter compiler than the project has ever seen. Velvet carried
a member template whose parameter pack shadowed the class pack -- accepted by MSVC, rejected
by Clang and GCC. Expect missing `typename`/`template` disambiguators, two-phase lookup
failures and narrowing conversions. Fixes are additive and arch-independent, so they help
the CUDA build too and are not HIP-specific hacks. (Velvet)

## Types, dispatch and platform limits

**Library swaps.** cuBLAS -> hipBLAS, cuFFT -> hipFFT, cuRAND -> hipRAND, cuSPARSE ->
hipSPARSE, cuDNN -> MIOpen, Thrust/CUB -> rocThrust/hipCUB. Mostly 1:1; watch handle types
and a few signature differences such as the hipBLAS v2 enums.

**Negative entries for that table.** CUPTI has no ROCm analogue -- stub the optional
per-kernel profiling path inert and let the timer fall back to `hipEvent` wall-clock rather
than substituting something. ck_tile's fused MHA ships headers but no prebuilt instance
library in `/opt/rocm/lib`, so `fmha_fwd` is declared-only and using it means vendoring CK's
codegen. (cuSZ, lc0)

**Never name the pointee struct of an opaque CUDA handle.** `CUstream_st`/`CUevent_st` do
not exist under HIP. Spell it `std::remove_pointer_t<cudaStream_t>` so it survives the
typedef swap. (cuSZ)

**`char` vs `signed char` vector base types differ.** HIP defines
`using char4 = HIP_vector_type<char, N>` with plain `char` members; NVIDIA's `char4` has
`signed char` members, and the two are distinct C++ types even when identically
represented. A type-machinery layer must report `char` as the base type of the canonical
`charN` on HIP, so a reference accessor binds. A test asserting
`is_same_v<signed char, BaseType<MakeType<signed char,N>>>` then fails on HIP. This is
dictated by the upstream HIP header, not the porting layer: forcing it would break the
reference accessor or introduce a non-canonical vector family into shared code. Numeric
behaviour is identical -- document it as an upstream type-identity deferral, not a bug.
(CV-CUDA: TypeTraitsMakeTypeVectorTest/3.)

**AMD compute-capability values COLLIDE with NVIDIA arch numbers**, so CC-keyed dispatch
picks the NVIDIA path at RUNTIME on AMD. `hipDeviceAttributeComputeCapabilityMajor` and `hipDeviceProp_t::major` report 9 on gfx90a --
the same value as Hopper sm90 -- so code selecting a kernel config by "cc>=9 means Hopper
DPX/wgmma" tries to launch intrinsics AMD lacks, and a green compile does not catch it.
Gate arch-specific selection on `__HIP_PLATFORM_AMD__` and force the portable config; never
trust CC major/minor for arch dispatch on AMD. Found independently by both libmarv ports.
(MMseqs2, foldseek: the CUDASW++ short2-DPX Smith-Waterman selector.)

**Runtime PTX plus the CUDA Driver API is a THIRD build class** beyond Strategy A and B.
Some projects never compile `.cu` into the binary -- they emit PTX text, embed it, and load
it at runtime via `cuModuleLoadData`/`cuLaunchKernel`, often through a hand-written FFI
binding. Port to the HIP Driver API: compile each `.cu` to a code object with
`hipcc --genco --offload-arch=<arch>` (not PTX); embed the bytes length-delimited and
binary-safe (base64) keyed by `gcnArchName`, suffix-stripped, NOT by "highest compute
capability"; pass `hipModuleLoadData` a pointer to the byte BUFFER, since a C-string
conversion truncates at the first NUL; select by `gcnArchName` with a trial-load fallback;
and launch via `hipModuleLaunchKernel`. Two cgo-specific traps: `hiprand.h` does
`typedef __half half` and AMD's `__half` is C++-only, breaking the C parse, so alias
`typedef _Float16 __half;` in the cgo preamble (it must sit in the comment block touching
`import "C"`); and `hipCtxSynchronize`/`hipCtxGetApiVersion` return `hipErrorNotSupported` on ROCm 7.2.1, so
synchronize via `hipStreamSynchronize`. (mumax3: Go + cgo + runtime code-object loading.)

**Compute-only CDNA has no graphics pipeline**, so OpenGL/Vulkan-interop apps build but
cannot run on gfx90a. Datacenter parts have no display engine: radeonsi refuses a GL context
on a compute chip, only software GL is available, and a software-GL buffer cannot be
registered with HIP. Anything sharing a buffer via `cudaGraphicsGLRegisterBuffer`/`hipGraphicsGLRegisterBuffer` fails at
registration with `hipErrorInvalidValue`, and kernels gated behind it are unreachable. The
build links cleanly, which is the trap. Mark gfx90a non-viable and scope it out of the PR
claim rather than recording a build-only pass; such a port runs on graphics-capable RDNA
only. The exception is a headless-compute refactor decoupling the solver from interop --
a larger port, rarely worth it for an interactive renderer. (Velvet: XPBD cloth solver
writing particle positions into GL VBOs.)

**An arch-specific fix keyed on the OS or the ROCm version can BREAK an already-validated
arch**, because in this fleet those are PROXIES for the real dependency. Windows hosts run
TheRock nightly wheels -- a newer ROCm but sometimes an OLDER torch -- while Linux runs an
older ROCm with a newer torch, so OS, ROCm version and torch hipify generation are tangled
and can ANTI-correlate. A guard written `#ifdef _WIN32` when the true dependency is the
hipify generation works only by that accidental correlation. Guard on the real axis, and
there are three:

- **OS and toolchain** -> `_WIN32`: MSVC-vs-MinGW linkage, DLL import libs, path
  separators. (The `/ALTERNATENAME` workaround for missing inherited-constructor exports
  belongs here; the reproducer is in the local `findings/` area, and the project's own
  `deferred.json` records whether it was filed.)
- **HIP runtime version** -> `HIP_VERSION_MAJOR`/`MINOR`: header behaviour, added or
  removed symbols. (MMseqs2: a Windows commit defined `HIP_DISABLE_WARP_SYNC_BUILTINS`
  unconditionally because `<amd_hip_bf16.h>` defines `__shfl_*_sync` on ROCm 7.14+, but on
  7.2.x `__syncwarp` lives inside the SAME guard, so the define stripped `__syncwarp` and
  broke Linux.)
- **torch hipify generation** -> `torch.utils.hipify.__version__`. v2 STOPS renaming the
  `c10`/`at` `cuda` classes to `hip`: the CUDA spellings stay public and the hip-spelled
  symbols become `USE_ROCM`-only, where v1 renamed the other way. In a torch CUDAExtension
  the TU IS hipified, so just write the CUDA spelling. In a NON-hipified TU -- a
  CMake/`USE_HIP` port, or a host `.cpp` routed around hipify -- detect the version at build
  time, pass a NEUTRAL define (`-DTORCH_HIPIFY_V2`, never a moat-named one), and branch:
  v2 -> `c10::cuda::getCurrentCUDAStream`, v1 -> `c10::hip::getCurrentHIPStream`. (aihwkit: a Windows commit hard-coded
  `c10::hip::getCurrentHIPStream`, needed on TheRock's older torch but absent on Linux's
  newer one, breaking that build.)

Validator corollary: a Windows-LABELED delta is NOT automatically inert elsewhere -- it can
touch a shared header that compiles differently against another platform's ROCm or torch.
When such a delta flips a passed arch to revalidate, REBUILD it rather than carrying forward
on "it is Windows-gated" reasoning. A binary-equivalence carry-forward inherently builds,
which is how both regressions above were caught; a reasoning-only skip would have shipped a
broken build. (aihwkit, MMseqs2)
