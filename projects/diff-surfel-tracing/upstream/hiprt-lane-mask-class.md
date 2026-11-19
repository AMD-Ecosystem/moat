<!--
repo: GPUOpen-LibrariesAndSDKs/HIPRT
title: BVH build hangs or silently drops geometry on 64-wide wavefronts (lane masks built with 32-bit literals)
note: 64-lane values computed in 32 bits break the BVH build on CDNA
-->

## Summary

Three places in `hiprt/impl/BvhBuilderKernels.h` compute a value that describes all 64 lanes of a wavefront using 32-bit arithmetic. The shift's result type is the promoted **left** operand, so a literal `1`, `1u` or `1 <<` produces a 32-bit value that is only then widened to 64 bits. On a 64-wide wavefront the upper half of the mask is wrong.

All three are invisible on RDNA, where a 32-lane wavefront never reaches a shift count of 32. That is consistent with HIP RT being validated on RDNA.

There is a fourth, structural item at the end of this report: the wavefront width is taken from an enumerated architecture list rather than from the compiler, which is what keeps producing this class of defect.

## 1. `subwarpMask` built in 32 bits (root cause of the BVH build hang)

`BvhBuilderKernels.h` lines 144, 709 and 1104 (`openNodes`, `FitBounds`, `Collapse`):

```c++
const uint64_t subwarpMask = ( ( 1 << BranchingFactor ) - 1 )
                             << static_cast<uint64_t>( ( BranchingFactor * subwarpIndex ) );
```

`( 1 << BranchingFactor ) - 1` has type `int`. The `static_cast<uint64_t>` on the **right** operand does not change the result type, so the whole expression evaluates in 32 bits. With `BranchingFactor` 4 and `WarpSize` 64, `subwarpIndex` runs 0..15 and the shift count reaches 60. Shifting an `int` by 32 or more is undefined, and the hardware shift uses only the low 5 bits of the count, so the count wraps and **lanes 32..63 receive the mask of lanes 0..31**.

`openNodes` then selects the widest child of a subgroup with

```c++
__ffsll( ballot( maxArea == area ) & subwarpMask ) - 1
```

so the upper half of the wavefront reads the lower half's ballot bits and opens a subtree belonging to a different task. The same subtree can be opened into two slots.

**This is the root cause of a GPU hang that is easy to misdiagnose.** Because `Collapse` now emits more leaf references than there are primitives, its exit test

```c++
atomicAdd( &header->m_referenceCount, 0 ) == referenceCount
```

is a monotonic counter compared for equality, so once the emitted count passes `referenceCount` the test is missed permanently and every lane spins forever in `while ( hiprt::any( !done ) )`.

Relaxing that comparison to `>=` makes hanging scenes build and trace, which is how we first "fixed" it -- but that only treats the symptom. `referenceIndices` is allocated with exactly `primitives.getCount()` entries, so the overshoot is also a real out-of-bounds write, and `>=` alone converts a hang into silent corruption. The mask is the defect; the exit test is where it becomes visible.

Observed on gfx90a and gfx942: a BVH build over a few dozen primitives hangs the GPU. It is data dependent rather than size dependent (44 surfel disks builds; 42 and 46 do not), and it affects the LBVH, LBVH-without-triangle-pairing and PLOC builders alike, because all three call `Collapse` and `openNodes`.

Fix, at all three sites:

```c++
const uint64_t subwarpMask = ( ( 1ull << BranchingFactor ) - 1ull )
                             << static_cast<uint64_t>( ( BranchingFactor * subwarpIndex ) );
```

Bit-for-bit identical on a 32-wide wavefront.

## 2. `PairTriangles` clears the wrong half of `activeMask` (silently drops half the geometry)

`BvhBuilderKernels.h:405`:

```c++
activeMask &= ~( 1u << firstPairedLane );
```

`~( 1u << ... )` is an `unsigned int`, so it zero-extends into the 64-bit AND and clears bits 32..63 of `activeMask` for every lane, whatever the shift count. The loop that gives each lane its turn as the broadcast lane therefore stops as soon as the low 32 bits drain. Lanes 32..63 keep `pairedIndex == InvalidValue` and their triangles are never written to `pairIndices`.

**Half of every 64-wide wavefront's triangles are absent from the BVH.** The build succeeds, there is no error and no hang, and the rendered image looks plausible. The loss is only visible if something counts the primitives actually traced.

Measured on gfx90a: **31 of 64 surfels received a gradient before the fix, 62 of 64 after it.**

Fix:

```c++
activeMask &= ~( 1ull << firstPairedLane );
```

## 3. `PackLeavesWarp` sets bits it cannot clear (hang on RTIP 3.1 and later)

`BvhBuilderKernels.h:1989` and `:1993`:

```c++
packetMask ^= 1 << broadcastLane0;
if ( secondValid ) packetMask ^= 1 << broadcastLane1;
```

`1` is a signed `int`, so at lane 31 the result is `INT_MIN` and sign-extends to `0xffffffff80000000` when xored into the 64-bit `packetMask`, setting bits 32..63 that the enclosing `while ( packetMask )` can never clear.

`PackLeavesWarp` builds `TrianglePacketNode`, which is selected for RTIP 3.1 and later (gfx1201). **We want to be clear that this one is reasoned from the source and not measured** -- no gfx1201 hardware was available to us -- but it is the same defect shape as the two above.

Fix:

```c++
packetMask ^= 1ull << broadcastLane0;
if ( secondValid ) packetMask ^= 1ull << broadcastLane1;
```

## 4. The structural item: wavefront width comes from an architecture allowlist

`hiprt/hiprt_common.h:202-206` sets `constexpr uint32_t WarpSize = 64` for an enumerated list of architectures (`__gfx900__`, `902`, `904`, `906`, `908`, `909`, `90a`, `90c`, `940`, `941`, `942`) and 32 for everything else.

gfx950 (MI350) is a 64-lane architecture that is **not** on that list. HIP RT's runtime-compiled kernels would be built there with `WarpSize = 32` against 64-lane hardware, so `laneIndex = threadIdx.x % WarpSize` stops being the lane index, and every subgroup carve, lane mask and ballot reduction downstream of it -- `openNodes`, `FitBounds`, `Collapse`, `PairTriangles`, `PackLeavesWarp` -- is computed against the wrong half of the wavefront.

We have not patched this locally and deliberately so: we have no gfx950 to validate against, and adding `__gfx950__` to the list would perpetuate the enumeration rather than fix it. The durable fix is to take the width from the compiler (`__AMDGCN_WAVEFRONT_SIZE__`, or `warpSize`), which is also what stops the next architecture from re-introducing items 1 to 3.

This item is reasoned rather than measured, for the same reason it matters: gfx90a and gfx942 are both on the allowlist, so testing on them cannot see it.

## Versions checked

Tag `3.1.0.cb09c56` (commit `8602b8c475255fb922c2792654aae0a6bcdeb0af`). All four items were re-checked against HIP RT HEAD `e3c01fc` on 2026-08-14 and are unchanged there, so none of this is staleness in the pinned tag.

Items 1 to 3 are fixed locally and the fixes are arch-unified by construction -- every changed expression is bit-identical on a 32-wide wavefront. We are happy to open a pull request for those three, and to discuss item 4 separately if you would prefer to keep the allowlist.
