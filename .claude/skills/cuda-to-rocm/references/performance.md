# Performance: what a correct port can still lose

A port that is byte-identical to CUDA can still be an order of magnitude slower on AMD. The
causes below were measured, not guessed: each was swapped in and out one at a time on the
same binary and input. Check them before calling a port done, and before blaming tile
tables or wavefront size.

## How to find the cost in an hour

1. `rocprofv3 --kernel-trace --stats -d <dir> -o run --output-format csv -- <cmd>` gives
   per-kernel time and call counts (`run_kernel_stats.csv`). rocprofv3 aborts if
   `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` are both set, so unset the CUDA one.
2. Pull the code objects out of the binary (`llvm-objdump --offloading <bin>`, which writes
   `<bin>.N.hip-amdgcn-amd-amdhsa--gfxXXXX` next to it) and read
   `llvm-readelf --notes` for each hot kernel: `.vgpr_count`, `.vgpr_spill_count`,
   `.private_segment_fixed_size` (scratch). Then `llvm-objdump -d` the hot kernel and count
   instructions in its inner loop. Compare against a reference build (upstream's own HIP
   path, or the CUDA path's intent) at the same kernel.
3. Swap one difference at a time. When the hot code lives in a shared library, build only
   that target per variant and point `LD_LIBRARY_PATH` at it; `DT_RUNPATH` yields to it.

## Scalar emulation of a packed intrinsic

CUDA has `__hmax2`/`__hmin2` for `half2`; HIP (checked through HIP 7.16) has only the
scalar `__hmax`/`__hmin`, and ROCm's scalar `__hmax` is three explicit `__hisnan` tests
plus a compare. Emulating `__hmax2` as two scalar `__hmax` calls is correct but the
compiler keeps the NaN tests as compares, selects and `s_and_saveexec` branches instead of
one `v_pk_max_f16`. In a max-heavy inner loop that is roughly 14x more instructions per
iteration, and the bloat then pushes `#pragma unroll` loops over the unroll threshold, so
per-thread register arrays land in scratch.

Use the compiler's packed builtin, which lowers to `v_pk_max_f16` and keeps CUDA's
non-NaN-operand semantics (`llvm.maxnum`):

```
typedef _Float16 h2v __attribute__((ext_vector_type(2)));
return __builtin_bit_cast(__half2, __builtin_elementwise_max(
    __builtin_bit_cast(h2v, a), __builtin_bit_cast(h2v, b)));
```

Do not use `asm volatile("v_pk_max_f16 ...")`. It emits the right instruction, but
`volatile` pins every asm block in program order, so the scheduler cannot hoist LDS loads
ahead of them and the loop waits on each load. Plain `asm` (no `volatile`) recovers most of
it; the builtin is fastest and portable across gfx9/gfx11/gfx12.

Measured (MMseqs2 `ungappedprefilter`, 1000 queries, gfx1100, identical output in every
row): scalar `__hmax` emulation 328 s; `asm volatile` 19.4-21.0 s; non-volatile `asm`
11.4 s; `__builtin_elementwise_max` 9.2-10.5 s. The same pattern applies to any SIMD-in-word
CUDA intrinsic emulated lane by lane (`__vmaxs2`, `__viaddmax_*`, `__vadd4`): check whether
a packed AMD instruction or an `__builtin_elementwise_*` form exists before writing scalar
lanes, and look at the ISA of the result either way.

## Tile tables are second order

Pinning a conservative NVIDIA tile table (T4/sm75) instead of an AMD-tuned one cost about
10% once the arithmetic was fast, and the "AMD-tuned" table was the slower of the two on
RDNA3. With the slow scalar emulation the tuned table was 37% slower, because its larger
per-thread tiles spill. Fix the instruction stream first, then tune tiles with the profiler.

## What did not matter

A SHARED library with `-fgpu-rdc` versus a static library with CMake's device link: same
time within noise for the same source. Redirecting `__shfl_*_sync` to maskless shuffles had
no effect because the hot kernels use cooperative-groups tile shuffles, which compile to the
same `ds_bpermute_b32` either way.
