<!--
repo: GPUOpen-LibrariesAndSDKs/Orochi
title: Radix sort corrupts output and faults the GPU on 64-wide wavefronts (OnesweepReorder ballot truncated to 32 bits)
note: OnesweepReorder ranks half the block against the wrong lanes on a 64-wide wavefront
-->

## Summary

`OnesweepReorder` in `ParallelPrimitives/RadixSortKernels.h` treats a 256-thread block as eight logical warps of `WARP_SIZE` lanes, where `WARP_SIZE` is 32 (`RadixSortConfigs.h:45`). It then ranks keys using `__ballot()`, which reports **all 64 lanes of the hardware wavefront** on CDNA. The result is stored in a `u32`, which keeps lanes 0..31 and discards the caller's own half.

The consequence is that every odd logical warp ranks its keys against the even warp's attendee set. `warpOffsets` and `lpSum` are wrong for half the block, and those prefix sums index the global `outputKeys` and `outputValues` writes.

On a 32-wide wavefront the two halves coincide, so this is invisible on RDNA.

## Affected code

`ParallelPrimitives/RadixSortKernels.h`, around lines 465 and 476:

```c++
u32 broThreads = __ballot( itemIndex < numberOfInputs );          // keeps lanes 0..31 only

u32 difference = ( 0xFFFFFFFF * bit ) ^ __ballot( bit != 0 );     // same truncation
```

## Impact and reproduction

Measured on gfx90a (MI250X) driving this kernel through HIP RT's LBVH builder.

Any sort larger than `SINGLE_SORT_WG_SIZE * SINGLE_SORT_N_ITEMS_PER_WI` = 3072 keys dies with:

```
Memory access fault ... kernel: OnesweepReorderKeyPair64
```

Below 3072 keys, `RadixSort.cpp:243` selects the single-pass kernel and `OnesweepReorder` never runs, which is why only realistically sized workloads hit it. This affects any Orochi user sorting more than 3072 elements on a 64-wide wavefront.

## Suggested fix

Read the calling lane's own logical warp out of the wavefront ballot. This is bit-identical on a 32-wide wavefront (the shift is zero) and on the CUDA `__ballot_sync` path, so it needs no architecture branch at the call sites.

```c++
__device__ inline u32 logicalWarpBallot( bool predicate )
{
#if defined( ITS )
	return __ballot_sync( 0xFFFFFFFF, predicate );
#else
	const u32 logicalWarpInWave = ( threadIdx.x % warpSize ) / WARP_SIZE;
	return static_cast<u32>( __ballot( predicate ) >> ( logicalWarpInWave * WARP_SIZE ) );
#endif
}
```

Both call sites then become `logicalWarpBallot( ... )`.

## Version checked

Present at Orochi HEAD `78fb3df` (checked 2026-08-13). The same blob (`3fe3729`) is vendored byte-identically into HIP RT at tag `3.1.0.cb09c56` and is still identical at HIP RT HEAD `e3c01fc`, so HIP RT picks the fix up by re-vendoring.

We are happy to open a pull request with the above if that is useful.
