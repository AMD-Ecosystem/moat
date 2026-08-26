# cubvh round-2 perf follow-up on linux-gfx1100

Status: EXECUTED 2026-08-26 on linux-gfx1100 (W7800); results in notes.md "Validation (round 2, linux-gfx1100)" Parts A-D.
Author: validator, windows-gfx1151, 2026-08-26.
Target head: `c7379c0` (round-2 clean-room rewrite), baseline `e5a657a`.

## What is already settled, and what is not

windows-gfx1151 found a reproducible regression in the round-2 rewrite --
`sdf_raystab` +21-27% and `ray_trace` +6-12% at 2,000,000 triangles -- and
root-caused it to VGPR pressure halving occupancy on exactly the two
ray-traversal kernels. See notes.md, "Perf regression confirmed and
root-caused" and "Fix proven at codegen and runtime".

Two questions came out of that. **One is already answered and does not need
gfx1100 hardware**, because VGPR count is a compile-time property of source
plus target: gfx1151 cross-compiled `c7379c0` and `e5a657a` for gfx1100 and
gfx90a and read the register counts straight out of the code objects.

| arch | ray_trace VGPR | sdf_raystab VGPR | waves/SIMD (old -> new) |
|---|---|---|---|
| gfx1151 (RDNA3.5, wave32) | 55 -> 172 | 85 -> 171 | 16 -> 8 |
| **gfx1100 (RDNA3, wave32)** | **52 -> 172** | **85 -> 171** | **16 -> 8** |
| gfx90a (CDNA2, wave64) | 63 -> 86 | 98 -> 114 | 8 -> 5, 5 -> 4 |

So the codegen question is closed: **the blow-up is not gfx1151-specific.**
gfx1100 is identical to the register. gfx90a is affected far less in absolute
terms, which is consistent with the porter's wave64 measurements looking
clean.

What is NOT settled, and what this plan is for:

> **How much runtime damage does the occupancy loss actually do on a discrete
> wave32 part with a real memory system?**

gfx1151 is a 20-CU APU on shared LPDDR5. Occupancy buys latency hiding, and
how much that is worth depends entirely on the memory system. gfx1100 is
wave32 like gfx1151 but 60 CUs with dedicated GDDR6. It is the one platform
that separates "wave32 codegen problem" from "APU bandwidth problem", and the
answer decides whether this is a blocker or a disclosable footnote.

**It is also nearly free.** linux-gfx1100 is already stale (`validated_sha`
e5a657a, head c7379c0), so it owes a revalidation regardless, and that
revalidation must build BOTH shas anyway -- exactly the pair the A/B needs.
Marginal cost over work already owed: roughly 30-40 minutes of bench runs.

## Predictions to falsify

1. gfx1100 WILL show 172/171 VGPRs on the two kernels at `c7379c0`
   (already cross-compiled here; this is just confirmation on the real host).
2. gfx1100 WILL regress on `sdf_raystab` and `ray_trace` at big2m.
3. The magnitude will be SMALLER than gfx1151's +27%, because GDDR6 hides
   latency better than shared LPDDR5.

If (3) holds and the penalty is small (say under 8%), a person may reasonably
accept and disclose. If gfx1100 regresses as hard as the APU, the fix below
should land before the rewrite is published. If (1) fails on real hardware,
something is wrong with this analysis and it should be re-opened.

## Preconditions

- Branch `port/cubvh`, pulled. Selector should report `linux-gfx1100` as
  `revalidate`.
- Do NOT take the `porting` lock. Validation is unlocked and per-platform.
- Fork clone at `projects/cubvh/src`, `moat-port` at `c7379c0`, submodule
  `third_party/eigen` initialised.
- A ROCm torch venv. gfx1151 used torch 2.12.0+rocm7.14; match what the host
  normally uses and record it.
- Harness deps beyond torch: `trimesh`, `rtree`, `scipy`, `rich`, `pytest`.
  Without `rtree` and `scipy` the distance tests abort inside trimesh's CPU
  baseline before reaching a single assertion, which looks like a port
  failure and is not.

## Part A -- the revalidation that is already owed

Do this first and completely; it is the part with standing.

1. `PYTORCH_ROCM_ARCH=gfx1100`. Check out `e5a657a` detached, build clean,
   capture this platform's goldens:
   `python harness/golden.py capture --out agent_space/goldens-e5a657a-gfx1100`
   The rewrite changed device code, so `classify` returns mixed/differ and
   there is no binary-equiv carry-forward. **Keep this build** -- Part C
   needs it staged, and rebuilding it later wastes the whole point.
2. Back to `moat-port` (`c7379c0`), build clean.
3. Both round-2 differential gates against that reference:
   `golden.py check --ref ...` and `golden.py crossload --ref ...`
4. The upstream suite: `test/signed_distance.py`, `test/unsigned_distance.py`,
   `test/state_dict.py`, `pytest test/cuhashtable.py`, `test/hashtable.py`
   (CPU), `test/sparse_voxel.py <mesh>`.
   Expected and pre-existing -- do not chase:
   - `unsigned_distance.py` fails its SECOND (cpoint, line 135) assertion
     roughly 1 in 3-8 runs, from unseeded `torch.randn` tie-breaking. The
     FIRST (distance) assertion must pass every time; that is the real gate.
     Run it 3+ times and record the ratio.
   - `sparse_voxel.py` ends in `ValueError: Both events must be recorded`
     (the script never calls `end.record()`). The `.npz` must still be
     written, and the active voxel count must be **4,455,224** on a
     unit-sphere fixture at res=1024 -- that value now holds on gfx90a,
     gfx1100, gfx1101, gfx1201 and gfx1151, so anything else is a real
     finding.
   - `state_dict.py` passes on Linux; the error-32 failure is Windows-only.
5. `python3 utils/jargon.py --port cubvh` clean, fork tree clean, then
   `set-state cubvh linux-gfx1100 completed` and `commit-project`.

Part A stands alone. If Part C is skipped or runs out of budget, the
revalidation still lands and wave32 evidence at `c7379c0` doubles.

## Part B -- confirm the register counts on the real host (minutes)

1. From each build's `_cubvh*.so`, extract the gfx1100 code object.
   `agent_space/ab/extract_co.py` (written for this, on port/cubvh) parses the
   embedded `__CLANG_OFFLOAD_BUNDLE__` and works unchanged on an ELF `.so`.
   `roc-obj-ls` / `roc-obj-extract` also work if present.
2. `llvm-readelf --notes <code object>`, read `.vgpr_count`,
   `.sgpr_count`, `.vgpr_spill_count`, `.private_segment_fixed_size`.
3. Pair by hand -- **the rewrite renamed every kernel**:
   | old | new |
   |---|---|
   | `raytrace_kernel` | `bvh_ray_trace_kernel` |
   | `signed_distance_raystab_kernel` | `bvh_signed_distance_raystab_kernel` |
   | `unsigned_distance_kernel` | `bvh_unsigned_distance_kernel` |
   | `signed_distance_watertight_kernel` | `bvh_signed_distance_watertight_kernel` |
4. Expect 52 -> 172 and 85 -> 171. Sanity check: the `api_gpu.cu` code object
   must be byte-identical between the two builds, since the rewrite did not
   touch it.

## Part C -- the A/B benchmark

**Method matters; two traps cost real time on gfx1151.**

1. **Stage both builds side by side; do not rebuild between runs.** The
   `cubvh` python package is byte-identical across the two shas
   (`git diff e5a657a c7379c0 -- cubvh/` is empty), so only the compiled
   extension differs:
   ```
   agent_space/ab/old/   <- copy of cubvh/ + the e5a657a _cubvh*.so
   agent_space/ab/new/   <- copy of cubvh/ + the c7379c0 _cubvh*.so
   ```
   Select with `PYTHONPATH=agent_space/ab/<side>:<repo>/projects/cubvh/harness`.
   This is why Part A step 1 says keep the e5a657a build.
2. `python harness/bench.py --out <json>` (500k queries, 5 reps, three scales).
3. **Run both orderings.** 3 rounds old-then-new, then 3 rounds new-then-old.
   The first gfx1151 pass ran only old-then-new and could not rule out a
   thermal/clock bias against whichever build was measured second. Reversing
   it is what turned "probably real" into confirmed.
4. **Report min-of-runs, not just the median.** On gfx1151 the `build` op is
   bimodal from host CPU contention -- old [70.9, 71.8, 118.3] vs new
   [66.0, 108.4, 110.2] -- and the medians landed on different modes,
   manufacturing a fake +51% regression on an op that is actually 7-9%
   FASTER. `build` is host-side BVH construction and the most exposed to
   this. Treat any large `build` delta as suspect until min-of-runs agrees.
   Quiesce the box if you can.
5. A result counts only if the sign holds in BOTH orderings and the per-round
   values do not overlap.

## Part D (optional) -- confirm the fix on wave32 hardware

gfx1151 already proved the fix works, at codegen and at runtime, as a
throwaway local patch (reverted, never committed). One attribute per kernel,
no algorithm change:

```c
__global__ __attribute__((amdgpu_waves_per_eu(16))) void bvh_ray_trace_kernel(...)
__global__ __attribute__((amdgpu_waves_per_eu(16))) void bvh_signed_distance_raystab_kernel(...)
```

gfx1151 result: VGPR 172 -> 66 and 171 -> 88, occupancy 8 -> 16 waves, **zero
spills**, and big2m `sdf_raystab` +27.5% -> +4.4% while big2m `ray_trace`
+12.2% -> -4.4% (now faster than the old code).

Note `__launch_bounds__(128, 4)` is a NO-OP here: on HIP the second argument
is min waves per EU, and 4 was already satisfied at 8 waves. Use
`amdgpu_waves_per_eu` or pass a demanding second argument.

If Part C shows gfx1100 regressing, repeating this patch there tells us
whether one attribute value serves both wave32 parts or whether the target
needs tuning per arch. Keep it a throwaway: reverting before completion is
required, because committing it moves `head_sha`.

## What to record

Append to `projects/cubvh/notes.md` under a dated heading, mirroring the
gfx1151 section so the three platforms read side by side:

- host, GPU, CU count, ROCm/torch versions, wavefront width;
- the VGPR/SGPR/spill/scratch table from Part B;
- the old-vs-new perf table for both orderings, min-of-3 and median-of-3;
- an explicit verdict on predictions 1-3 above -- that is the decision input;
- raw bench JSONs kept in `agent_space/` and named in the notes.

Then update the deferred item `cubvh-wave32-raystab-perf-regression` so the
ruling rests on two wave32 data points instead of one.

## Out of scope

Do not commit the Part D fix to `moat-port` from this plan. That is porter
work: it moves `head_sha` and forces every platform to revalidate, and the
attribute is AMD-specific so it must be guarded (`USE_ROCM` /
`__HIP_PLATFORM_AMD__`) or it breaks the CUDA no-regression gate. This plan
gathers evidence for that decision; it does not make it.
