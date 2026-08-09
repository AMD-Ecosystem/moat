# visionaray follow-up PR: hip::device_vector (PREPPED, HELD)

STATUS: **OPENED 2026-06-24 as szellmann/visionaray#54**
(https://github.com/szellmann/visionaray/pull/54), after anari-visionaray cleared
reviewer + validator on gfx90a (its sole consumer). Opened directly from
AMD-Ecosystem:moat-port (GitHub sees it ahead-by-1 since #51's content is in master); not
rebased, to avoid disturbing the moat-port tip anari-visionaray was validated against.
(The merged base-port PR #51 draft is in pr-draft.md; this is a separate follow-up.)

ON MERGE: repoint anari-visionaray's build/dep from AMD-Ecosystem/visionaray @ moat-port to
upstream visionaray (per [[moat-dep-submodule-repoint-after-merge]]), then close the
deferred item; advance-head will flip to revalidate but the identical tree carries
forward binary-equiv. Tracked in data/deferred.json as
`visionaray-device-vector-upstream-pr` (stays open until merged + repointed).

## Repo / base
- Upstream: szellmann/visionaray (follow-up to merged PR #51, which landed the base HIP/ROCm support).
- Commit: 421da19b on AMD-Ecosystem/visionaray @ moat-port, authored Jeff Daily.
- Branch mechanics on open: rebase the fork's `moat-port` onto current upstream `master`
  (drops the already-merged #51 commits, leaving just this one device_vector commit on a
  fresh base; moat-port is ~6 behind, none of those commits touch these files, so it
  rebases cleanly), push, open PR -> szellmann/visionaray master.
- Re-check before opening: refresh the base and re-run `gh pr list --repo szellmann/visionaray
  --state all --search "device_vector OR HIP OR ROCm"` for any new competing PR.

## Scope (vs upstream master): 2 files
```
 include/visionaray/hip/detail/device_vector.inl | +73/-7
 include/visionaray/hip/device_vector.h          | +15/-1
```

## Title
[ROCm] hip::device_vector: complete the host-side container interface

## Body
Follow-up to #51 (HIP/ROCm support). The HIP `device_vector` only implemented the methods the trivial-kernel tests exercise, so the GPU BVH builder path did not compile under HIP: constructing a `hip_index_bvh` from a host-built BVH (`build_top_down`) needs the host-side container interface that `cuda::device_vector` already provides. This adds the missing pieces and brings `hip::device_vector` to parity with `cuda::device_vector`:

- `reserve(size_t)` with growth semantics (new `capacity_` member), and `resize` refactored to use it
- `push_back` / `emplace_back(Args&&...)` / `clear`
- the templated `std::vector<T, A>` constructor and `explicit operator std::vector<T, A>()` (the BVH vectors use `aligned_allocator`, not the default allocator)

Mirrors `cuda::device_vector` one to one; CUDA and CPU paths are untouched. Header-only template change.

Test Plan:

Consumed by an in-progress anari-visionaray HIP device, which builds `hip_index_bvh` sampling BVHs for unstructured/AMR volume fields. Built that device for gfx90a against this header and rendered triangle-surface and structuredRegular-volume scenes headless on an AMD Instinct MI250X (gfx90a, ROCm 7.2); both produce nonzero, varied framebuffers matching the CPU device. The existing `hip_test` / `hip_random_test` still build and pass on gfx90a.

Authored with the assistance of Claude (Anthropic) as part of an effort to bring ROCm/HIP support to GPU-accelerated open source projects.
