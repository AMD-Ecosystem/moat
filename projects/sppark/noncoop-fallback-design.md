# sppark MSM: non-cooperative-launch fallback (design)

## Goal

Make sppark's Pippenger MSM run correctly on GPUs/runtimes that do not support
cooperative kernel launch (`hipLaunchCooperativeKernel` returns 719, device
`cooperativeLaunch == 0`). Root cause: the grid-wide barrier is built on Global
Wave Sync (GWS); RDNA4/Navi4 has no GWS hardware, and Windows ROCm under-reports
GWS on RDNA3 (ROCm/rocm-systems#401). The cooperative path is preserved unchanged
for systems that do have GWS (CDNA gfx90a, RDNA3-on-Linux gfx1100, all CUDA); the
fallback is additive and selected at runtime.

This feature does not need validation on GWS systems (the cooperative path there
is byte-for-byte the prior behavior). It is validated on no-GWS hardware
(gfx1201 RDNA4, gfx1101-on-Windows RDNA3).

## What depends on cooperative launch

In the `msm_t::invoke` path only two kernels use a grid-wide barrier
(`__ockl_grid_sync()` on HIP / `cooperative_groups::this_grid().sync()` on CUDA),
both launched via `stream_t::launch_coop`:

1. `sort` (msm/sort.cuh) -- a two-level radix sort. The grid barrier lives only in
   `upper_sort` (the cross-block top-level pass). `lower_sort` is per-block
   (`__syncthreads` only). The small-input branch of `sort_row` (single block,
   `blockIdx.x == 0`) uses no grid barrier at all.
2. `accumulate` (msm/pippenger.cuh) -- one grid barrier at the very end, guarding a
   reset of the persistent global work-queue counter `current` for the next launch.

`breakdown`, `integrate`, and the digit-list `batch_addition` overload used by the
MSM path are ordinary `<<<>>>` launches and need nothing. (The bitmap `add<>` /
`batch_diff` overload also has a grid barrier, but it is not reached by the MSM
correctness path; it gets the same treatment as `accumulate` for completeness.)

## Key enabling property

sppark already routes ALL cross-block communication through global memory: the
per-block digit histograms and the digit base offsets live in the global
`histogram[]` array, and the sorted output / temp live in global `inout[]` /
`temp[]`. The shared-memory `counters[]` array is per-block scratch that is fully
recomputed after each barrier from the global histogram. Therefore each
`grid.sync()` is a clean cut line: the state that must survive it is already in
global memory, and a kernel boundary on the same stream provides exactly the same
grid-wide ordering + memory visibility (in fact a stronger guarantee than
`grid.sync`, since it does not even require co-residency).

## Decomposition

### accumulate (easy)

The grid barrier only exists so that all blocks finish draining the persistent
`current` work-queue before thread 0 resets `current = 0`. `integrate` is already a
separate kernel, so the buckets written by `accumulate` are already fenced by the
kernel boundary. Fallback:

- Hoist `streams[]` from a function-local `static __device__` to a namespace-scope
  `__device__` array so a tiny reset kernel can see it.
- `accumulate` gains a `bool coop` template arg (or a sibling kernel) that drops the
  trailing `grid.sync()` + `current = 0` reset.
- Before each non-coop `accumulate` launch, zero `streams[sid]` with a
  `reset_current<<<1,1>>>(sid)` kernel on the same stream.

Accumulation math is byte-identical; only the launch wrapper and the
reset-placement differ.

### sort / upper_sort (the real work)

Split the persistent `upper_sort` megakernel at its two grid barriers into three
ordinary kernels, sharing the SAME per-phase device code via `__forceinline`
helpers (single source of truth; the cooperative `upper_sort` calls the same
helpers with `grid.sync()` between them, so its behavior is unchanged):

- `upper_partition(len) -> {base, slice}` -- the gridDim/blockIdx data partition
  (pure function of len, gridDim.x, blockIdx.x), recomputed by each phase.
- Phase 1 `upper_count_emit(...)`: count digits over this block's slice into shared
  `counters[]`, emit the per-block histogram to global `histogram[2 + (i<<digit) +
  blockIdx.x]`. (sort.cuh lines 139-159.)
- Phase 2 `upper_scan_scatter(...)`: recompute the cross-block exclusive prefix sum
  from the global histogram (warp-shuffle scan, requires `gridDim.x` a power of two
  and `<= WARP_SZ`), scatter src -> temp using the per-block offsets, and (block 0)
  finalize the global digit base offsets. (sort.cuh lines 168-245, verbatim.)

Non-cooperative kernels (2D grid preserved for the `blockIdx.y` window selection,
`lsbits` chosen by `blockIdx.y` exactly as the existing `sort`):

- `sort_upper_count   <<<{grid, gy}, blk, shmem>>>` -> phase 1
- `sort_upper_scatter <<<{grid, gy}, blk, shmem>>>` -> phase 2
- `sort_lower         <<<{grid, gy}, blk, shmem>>>` -> the existing `lower_sort` loop
  from `sort_row` (per-block; reads finalized histogram slices, writes sorted
  output).

The small-input branch (`sort_row` else, single block, no grid barrier) is reused
unchanged by launching the existing `sort` kernel with `gridDim.x == 1`. The
host-side two-level-vs-single decision mirrors `sort_row`'s condition exactly using
the real `grid_size` (NOT 1): `two_level = (wbits > DIGIT_BITS) || (lg2(grid_size)
&& wbits > lg2(grid_size) + 1)`.

All three kernels launch with the SAME `grid_size` as the cooperative launch so the
data partition and the `sub_laneid == blockIdx.x` scan mapping are preserved.

## Runtime dispatch

`msm_t` records `bool coop = gpu.props().cooperativeLaunch != 0` (overridable to
force the fallback for testing: env `SPPARK_FORCE_NONCOOP`). `digits()` and
`invoke()` branch:

- coop: existing `launch_coop(sort, ...)` / `launch_coop(accumulate, ...)`
  (unchanged).
- non-coop: the split-kernel sequences above, on the same stream `gpu[2]` /
  `gpu[i&1]` so all ordering is preserved.

CUDA is unaffected (`cooperativeLaunch` is 1 on all supported NVIDIA GPUs, so the
coop branch is always taken; the new kernels are dead code there).

## Testing

1. Force-fallback equivalence: `SPPARK_FORCE_NONCOOP=1` runs the existing
   `msm_correctness` (vs arkworks) through the fallback on ANY GPU. On a GWS GPU this
   directly proves coop-path == noncoop-path == arkworks. Reviewers can run it.
2. No-GWS integration: `msm_correctness` for bls12_381 (TEST_NPOW=15) and bls12_377
   (TEST_NPOW=14) must PASS on gfx1201 and gfx1101 (Windows ROCm 7.14), which today
   fail at the cooperative launch.
3. msm_correctness_sizes (added): a single test that sweeps npow in {4,8,10,12,14,16},
   each checked vs arkworks, so one run covers the small and large sort paths and the
   grid/slice remainder logic. Implemented in poc/msm-cuda/tests/msm.rs.

A standalone sort-kernel unit test (random digit arrays vs a CPU counting-sort
reference) was considered but not added: arkworks is a fully independent MSM reference,
and a sort defect would corrupt bucket assignment and fail msm_correctness, so the
end-to-end sweep across sizes/curves/two GPUs already exercises the sort fallback
transitively. It can be added later if isolated coverage is wanted.

## Upstream presentation

- One commit, `[ROCm]`-prefixed, additive: new kernels + a runtime branch; the
  cooperative path and all CUDA behavior are unchanged.
- Rationale in code comments (GWS / cooperativeLaunch==0), no MOAT jargon.
- The PR body notes the fallback is independently droppable: removing it leaves the
  cooperative-only MSM exactly as before.

## Status: IMPLEMENTED and validated (2026-06-12)

Implemented on AMD-Ecosystem/sppark moat-port @ 868aa18 (on top of validated 00cb1a7).
Final kernel names: sort_upper_count, sort_upper_scatter, sort_lower (sort.cuh);
reset_accumulate_counter + accumulate<...,COOP> (pippenger.cuh). Dispatch in
msm_t::launch_sort / launch_accumulate on gpu.props().cooperativeLaunch, with the
SPPARK_FORCE_NONCOOP override. Requires C++17 (the COOP guard uses `if constexpr`,
which guarantees the fallback kernel contains no GWS instruction).

Validated vs arkworks on gfx1201 (RDNA4) and gfx1101 (RDNA3-on-Windows), both
cooperativeLaunch==0: bls12_381 and bls12_377 msm_correctness + msm_correctness_sizes
PASS. See projects/sppark/notes.md for the full matrix.

Scope note: the bitmap add<> (batch_addition/batch_diff) cooperative overload is not on
the MSM correctness path and was left cooperative; it would need the same accumulate-style
treatment only if a no-GWS entry point that uses it is added.
