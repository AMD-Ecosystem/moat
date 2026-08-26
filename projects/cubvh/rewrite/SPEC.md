# cubvh BVH core: interface-and-behavior specification for independent rewrite

This document specifies the exact interfaces and observable behavior of five
files so they can be reimplemented independently from published algorithm
literature. It was written by a separate agent that analyzed the existing
code; the implementer works ONLY from this spec, the `keep/` snippets, and
the published references named below. The implementer must NOT open the prior
versions of the five files (in git history or elsewhere), any `*_hip.*`
build artifact, or the NVlabs/instant-ngp repository.

Files to write from scratch (complete replacements):

1. `include/gpu/common.h`
2. `include/gpu/triangle.cuh`
3. `include/gpu/bounding_box.cuh`
4. `include/gpu/bvh.cuh`
5. `src/bvh.cu`

Everything else in the repository stays untouched. The api layer
(`include/gpu/api_gpu.h`, `src/api_gpu.cu`) and `include/gpu/gpu_memory.h`
(the `GPUMemory<T>` RAII allocator: `.data()`, `.resize_and_copy_from_host(vec)`)
may be read freely and define the calling contract.

## Published references to implement from

- Ray-triangle intersection: Moller-Trumbore (1997), "Fast, Minimum Storage
  Ray/Triangle Intersection".
- Point-triangle distance: Eberly, "Distance Between Point and Triangle in
  3D" (Geometric Tools), or any standard edge/face-region method.
- AABB ray intersection: the slab method (Kay-Kajiya; see PBRT ch. 6).
- BVH construction: top-down median split on the axis of maximum centroid
  variance (PBRT ch. 4 "middle"/EqualCounts variants describe the family).
- Inside/outside by ray stabbing: Nooruddin & Turk, "Simplification and
  Repair of Polygonal Models Using Volumetric Techniques" (2003).
- Sphere point distribution: the offset Fibonacci lattice, R. Roberts,
  "How to evenly distribute points on a sphere more effectively than the
  canonical Fibonacci lattice" (extremelearning.com.au) -- the epsilon-offset
  construction, exact formulas below.
- Threaded (stackless) tree traversal via escape links: standard technique;
  any textbook description of "ropes"/escape indices.

Comment style: minimal, ASCII only. One-line citations to the sources above
where an algorithm starts are welcome. Do not mention any internal project
vocabulary. Do not add copyright headers (house style has none).

## Hard compatibility constraints (bit-level)

These are serialization/ABI facts; they must hold exactly.

- `struct Triangle` POD layout, in this member order:
  `Eigen::Vector3f a, b, c; int64_t id;` (sizeof == 48, id at offset 40).
  Serialized by memcpy in the api layer; do not reorder or add members.
- `struct BoundingBox` POD layout: `Eigen::Vector3f min, max;` in that
  order (sizeof == 24). Do not reorder or add data members.
- `struct TriangleBvhNode` layout: `BoundingBox bb; int left_idx;
  int right_idx; int escape_idx;` (sizeof == 36, 9 int32 words).
  Node encoding (used by traversal AND by previously-serialized BVHs, which
  must still load and answer queries identically):
  - `left_idx < 0`: leaf. Triangle index range is
    `[-left_idx - 1, -right_idx - 1)` into the triangle array.
  - `left_idx >= 0`: inner node. Children are the contiguous node range
    `[left_idx, right_idx)`; with branching factor 4, `right_idx ==
    left_idx + 4` in newly built trees, but traversal must use the stored
    range, not assume 4.
  - `escape_idx`: index of the next node to visit after skipping this
    node's entire subtree, in pre-order; -1 means traversal is complete.
  - The root is node 0.
- `state_dict()["nodes"]` is an int32 CPU tensor of shape [n_nodes, 9]
  viewing the node array bytes; `["triangles"]` is [n_triangles, 12]
  viewing the Triangle array bytes (both produced by the KEPT code in
  `keep/bvh_state_dict_block.txt` and the api layer; you get this for free
  by preserving the POD layouts).

## Fixed behavioral constants

- `MAX_DIST = 1000.0f`, `MAX_DIST_SQ = MAX_DIST * MAX_DIST` (file-local to
  `src/bvh.cu`).
- Branching factor: 4. Leaf capacity: the `n_primitives_per_leaf` build
  argument (the api layer passes 8).
- Traversal stack capacity: 32 entries for ray and normal-averaging
  queries; 64 for closest-triangle queries.
- Kernel launch: 128 threads per block, `ceil(n/128)` blocks, on the
  caller-provided stream; skip the launch entirely when `n_elements <= 0`.
- Surface-coincidence epsilon for normal averaging: squared distance
  `< 1e-6f`.
- Ray-stab: 32 stab directions.
- Division guard epsilon (`safe_divide`, kept verbatim): 1e-6f.

## File 1: `include/gpu/common.h`

Namespace `cubvh`. Includes it must provide (downstream headers rely on
them transitively): `<iostream> <string> <vector> <cstdint> <cmath>
<cuda.h> <cuda_runtime.h> <cuda_fp16.h> <Eigen/Dense>`.

Required exports (names pinned -- `div_round_up` is used by
`include/gpu/hashtable.cuh`, which is not being rewritten):

- `constexpr float PI` (= 3.14159265358979323846f) and
  `constexpr float SQRT2` (= 1.41421356237309504880f). KEEP these names.
- `template <typename T> __host__ __device__ T div_round_up(T val, T divisor)`
  -- ceiling division.
- A kernel-launch helper for 1-D grids used by `src/bvh.cu`: 128 threads
  per block; the kernel's first parameter is `(uint32_t)n_elements`;
  a shared-memory-bytes argument and a `cudaStream_t`; no launch when
  `n_elements <= 0`. Free choice of name/signature since all callers are
  rewritten, but keeping the historical shape
  `linear_kernel(kernel, shmem_bytes, stream, n, args...)` is acceptable.
- `sign(float)` returning +-1.0f with `copysignf(1.0f, x)` semantics
  (sign of the FLOAT including signed zero), `__host__ __device__`.
- `fractf(float x)` = `x - floorf(x)`.
- `safe_divide` -- copy VERBATIM from `keep/common_safe_divide.txt`
  (it is original MIT code, not part of the rewrite).
- Small clamp/swap helpers as needed by your own code (host+device safe;
  do not rely on `std::` algorithms inside device code).

### Fibonacci sphere directions (exact behavior pinned)

`template <uint32_t N_DIRS> __host__ __device__ Eigen::Vector3f
fibonacci_dir(uint32_t i, const Eigen::Vector2f& offset)`

Implements the offset Fibonacci lattice from the Roberts article, mapped to
the sphere, with a per-call 2-D random offset (for randomized rotation of
the lattice). The exact math (required for cross-build determinism of the
ray-stab mode):

- epsilon by lattice size, per the article's table:
  N >= 11000 -> 27; N >= 890 -> 10; N >= 177 -> 3.33; N >= 24 -> 1.33;
  else 0.33. (All float.)
- golden ratio phi = 1.6180339887498948482045868343656f.
- u = fractf((i + epsilon) / (N_DIRS - 1 + 2*epsilon) + offset.x())
- v = fractf(i / phi + offset.y())
- Map the unit square to the sphere in cylindrical form:
  cos_theta = 1 - 2*u  (i.e. -2u + 1)
  sin_theta = sqrtf(fmaxf(1 - cos_theta^2, 0))
  angle = 2*PI*(v - 0.5f), evaluated with `sincosf(angle, &s, &c)`
  direction = { sin_theta * c, sin_theta * s, cos_theta }.

You may decompose this differently (e.g. a helper mapping (u,v) to a
direction), but the arithmetic above must be reproduced exactly, including
the `- 0.5f` phase and the `fmaxf` clamp.

## File 2: `include/gpu/triangle.cuh`

Namespace `cubvh`, includes `<gpu/common.h>`. Defines `struct Triangle`
with the pinned layout and these `__host__ __device__` members:

- `Eigen::Vector3f centroid() const` -> (a+b+c)/3.
- `float centroid(int axis) const` -> (a[axis]+b[axis]+c[axis])/3.
- `Eigen::Vector3f normal() const` -> unit normal in the direction of
  cross(b-a, c-a). For a zero-area triangle the result is the natural
  NaN vector from normalizing a zero vector (do NOT special-case it).
- `float ray_intersect(const Eigen::Vector3f& ro, const Eigen::Vector3f& rd) const`
  Double-sided ray-triangle intersection (Moller-Trumbore). Behavior:
  - Returns the ray parameter t >= 0 of the intersection point when the
    ray (origin ro, direction rd, not necessarily normalized) hits the
    triangle, including t == 0 (origin exactly on the triangle counts as
    a hit).
  - On a miss, a rejected barycentric range, t < 0, or a (near-)parallel /
    degenerate configuration, return the sentinel `1e6f`. Never return a
    negative value and never NaN/Inf: guard the determinant (reject when
    the reciprocal would blow up; a threshold on |det| relative to 1e-6f,
    or the kept `safe_divide`, both qualify -- pick one and be explicit).
  - Barycentric acceptance is the closed range: reject only when
    u < 0, u > 1, v < 0, or u + v > 1 (boundary hits accepted).
- `float distance_sq(const Eigen::Vector3f& p) const`
  Exact squared Euclidean distance from p to the (double-sided) triangle:
  the face-interior distance when the projection of p lies inside, else
  the distance to the nearest clamped edge segment. Any published
  region-based or three-edge-clamp formulation is fine.
  Degenerate-triangle rule: for a zero-area triangle the result must FAIL
  `<=` comparisons against any finite bound (NaN from a 0/0 in the face
  branch, or +infinity, both qualify). Do not return a finite segment
  distance for zero-area triangles -- selection behavior depends on them
  never winning a closest-triangle competition.
- KEEP VERBATIM from `keep/triangle_original_members.txt` (original MIT
  code, not part of the rewrite): `point_in_triangle`,
  `closest_point_to_line`, `closest_point`, `barycentric`.
- Do NOT reimplement or keep: uniform surface sampling, surface area,
  unsquared distance, vertex-array getter, stream output operator. They
  are unused; dropping them is deliberate.

## File 3: `include/gpu/bounding_box.cuh`

Namespace `cubvh`, includes `<gpu/common.h>` and `<gpu/triangle.cuh>`.
Defines `struct BoundingBox` with the pinned layout and:

- Default constructor: empty box (`min` = +infinity in all components,
  `max` = -infinity). `__host__ __device__`.
- Host constructor from a triangle range
  `(std::vector<Triangle>::iterator begin, std::vector<Triangle>::iterator end)`:
  the tight AABB over all three vertices of every triangle in [begin, end).
  (Callers never pass an empty range.)
- `void enlarge(const Eigen::Vector3f& p)` and
  `void enlarge(const Triangle& t)`: grow to include.
- `float distance_sq(const Eigen::Vector3f& p) const`: squared distance
  from p to the box, 0 when p is inside. The componentwise form
  max(min - p, p - max, 0) has one natural spelling; any equivalent is
  fine.
- `Eigen::Vector2f ray_intersect(<vector args>) const`: the slab method.
  Behavior pinned:
  - Per axis, the two slab parameters are computed with the KEPT
    `safe_divide(plane - origin, direction)` (epsilon 1e-6f) -- this
    guard's exact semantics matter for axis-parallel rays.
  - Swap the pair when needed so tmin <= tmax per axis, then intersect
    the three intervals.
  - As soon as the running interval is empty, return
    `{FLT_MAX, FLT_MAX}` (std::numeric_limits<float>::max()).
  - Otherwise return `{tmin, tmax}` WITHOUT clamping to zero: a box
    containing or behind the origin yields a negative tmin, and callers
    rely on that (a negative tmin always compares smaller than any hit
    distance, so such boxes are always descended into).
  - Parameter types: accept the origin and direction as
    `Eigen::Ref<const Eigen::Vector3f>` (the traversal passes both plain
    vectors and mapped rows).
- Do NOT keep: triangle-overlap (separating axis) test, box-box
  intersection/containment, signed distance, inflate, center, diag,
  relative position, corner enumeration, stream output. All unused;
  dropping them is deliberate.

## File 4: `include/gpu/bvh.cuh`

Namespace `cubvh`. Includes `<gpu/common.h>`, `<gpu/triangle.cuh>`,
`<gpu/bounding_box.cuh>`, `<gpu/gpu_memory.h>`, `<memory>`,
`<torch/torch.h>`.

- `struct TriangleBvhNode` exactly as pinned above (with a brief comment
  documenting the leaf/inner/escape encoding -- the encoding is a
  serialization contract, so document it here).
- `template <typename T, int MAX_SIZE = 32> class FixedStack`:
  fixed-capacity LIFO with `push(T)`, `T pop()`, `bool empty() const`,
  `bool overflowed() const`, all `__host__ __device__`.
  Overflow behavior: a push beyond capacity is DROPPED, sets a sticky
  overflow flag, and prints a one-time warning (printf; once per stack
  instance, not per push). `pop()` on an empty stack is undefined
  (callers guard with `empty()`).
  Aliases: `using FixedIntStack = FixedStack<int>;`
  `using FixedIntStackLarge = FixedStack<int, 64>;`
- `class TriangleBvh` (abstract):
  - protected: `std::vector<TriangleBvhNode> m_nodes;
    GPUMemory<TriangleBvhNode> m_nodes_gpu;` and a protected default
    constructor.
  - public pure virtuals, signatures verbatim:
    ```
    virtual void build(std::vector<Triangle>& triangles, uint32_t n_primitives_per_leaf) = 0;
    virtual void signed_distance_gpu(uint32_t n_elements, uint32_t mode, const float* positions, float* distances, int64_t* face_id, float* uvw, const Triangle* gpu_triangles, cudaStream_t stream) = 0;
    virtual void unsigned_distance_gpu(uint32_t n_elements, const float* positions, float* distances, int64_t* face_id, float* uvw, const Triangle* gpu_triangles, cudaStream_t stream) = 0;
    virtual void ray_trace_gpu(uint32_t n_elements, const float* rays_o, const float* rays_d, float* positions, int64_t* face_id, float* depth, const Triangle* gpu_triangles, cudaStream_t stream) = 0;
    ```
  - `static std::unique_ptr<TriangleBvh> make();`
  - `TriangleBvhNode* nodes_gpu() const { return m_nodes_gpu.data(); }`
  - KEEP VERBATIM the serialization block from
    `keep/bvh_state_dict_block.txt` (`state_dict()` /
    `load_state_dict()`; original MIT code, not part of the rewrite).

## File 5: `src/bvh.cu`

Namespace `cubvh`; `using namespace Eigen;`. One concrete subclass
(branching factor 4 as a template or constant -- your choice of
decomposition) plus four kernels and `TriangleBvh::make()` returning the
concrete type.

### build(triangles, n_primitives_per_leaf)

Top-down, in place: the build PARTITIONS (reorders) the caller's triangle
vector; leaves reference contiguous ranges of it. Algorithm:

- Clear `m_nodes`. Node 0 is the root; its box is the AABB of all
  triangles. The root is always split (the api layer guarantees > 8
  triangles).
- Splitting a range: choose the axis with maximum VARIANCE of triangle
  centroids (mean over the range, then componentwise squared deviations;
  ties resolved by whatever the max-coefficient search returns);
  partition around the median element with `std::nth_element` comparing
  `centroid(axis)` (median = begin + count/2). Applying this split twice
  (split each half again, each with its own axis choice) yields 4
  contiguous sub-ranges = the node's 4 children.
- Children occupy 4 CONTIGUOUS slots in `m_nodes`, appended together;
  parent's `left_idx` = first child slot, `right_idx` = one past the
  last. Each child gets its range's AABB. A child whose range has
  <= n_primitives_per_leaf triangles becomes a leaf (encode the range per
  the node encoding); otherwise it is split in turn.
- Ordering freedom: which pending node is split next (DFS recursion,
  explicit stack, queue...) is your choice -- any order is valid as long
  as sibling blocks are contiguous and indices are consistent. Node
  arrays from other builds (different order) must still traverse
  correctly, which the encoding guarantees.
- After all nodes exist, thread the escape links: `escape_idx` of a node
  = the next node in pre-order AFTER its subtree (its next sibling if
  any, else the parent's escape), -1 at the end; root's escape is -1.
  Children are visited in slot order.

### Traversal queries

All are `static __host__ __device__` on the concrete class (host
callable for debugging; device used by kernels). Two traversal schemes:
a fixed-stack scheme (fast path) and an escape-link (stackless) scheme
used as the overflow fallback and valid for ANY well-formed node array
(including deserialized ones).

Stack scheme common shape: push root index 0; pop; leaf -> scan its
triangle range; inner -> compute a key for each of the 4 children
(left_idx + i), push the candidates ORDERED so the most promising is
popped first (i.e. push in decreasing-key order). After processing each
node, if the stack has overflowed, abandon and return the stackless
fallback's answer (which restarts the query from the root).

1. `ray_intersect(ro, rd, nodes, triangles) -> std::pair<int, float>`
   (triangle ARRAY INDEX -- not id -- and ray parameter).
   - best t starts at MAX_DIST, best index at -1.
   - Leaf: for each triangle, `t = tri.ray_intersect(ro, rd)`; strictly
     smaller t wins (the miss sentinel 1e6 never beats MAX_DIST).
   - Inner: child key = `child.bb.ray_intersect(ro, rd).x()` (the slab
     tmin); prune children with key >= current best t (strict < required
     to descend); nearest-first ordering among the surviving pushes.
   - Overflow fallback: stackless restart with best t = MAX_DIST.
   - Stackless scheme: walk `idx = 0`; at each node, if the node's slab
     tmin >= current best t, jump to `escape_idx`; at a leaf, scan
     triangles then jump to `escape_idx`; at a surviving inner node,
     step to `left_idx` (first child; siblings are reached later via
     their escape links); stop at -1.
   - Signature note: accept ro/rd as `Ref<const Vector3f>`.
2. `closest_triangle(point, nodes, triangles, max_distance_sq = MAX_DIST_SQ)
   -> std::pair<int, float>` (array index, UNSQUARED distance).
   - Uses the 64-deep stack.
   - Best squared distance starts at `max_distance_sq`, best index -1.
   - Leaf: `d2 = tri.distance_sq(point)`; accept with `d2 <= best`
     (NON-strict: an exact tie updates to the LATER triangle in scan
     order; NaN d2 from degenerate triangles fails and is skipped).
   - Inner: child key = `child.bb.distance_sq(point)`; descend when
     `key <= best` (non-strict); nearest-first among pushes.
   - Overflow fallback: stackless restart passing the CURRENT best
     squared distance as the bound (the best index restarts at -1; with
     non-strict acceptance the same triangle is re-found).
   - If the final best index is -1 (nothing within the bound), return
     `{0, 0.0f}` -- a historical quirk that callers tolerate; preserve
     it in both schemes.
3. `avg_normal_around_point(point, nodes, triangles) -> Vector3f`
   (point is assumed to lie ON the surface).
   - 32-deep stack. Box pruning: descend a child only when
     `bb.distance_sq(point) < 1e-6f` (STRICT <; no ordering needed --
     order of accumulation may differ across schemes, which is
     acceptable float noise). Leaf: accumulate `tri.normal()` (unit
     normals, weight 1) for every triangle with
     `distance_sq(point) < 1e-6f`; count the weight.
   - Return (sum of normals) / (total weight): unnormalized average;
     0/0 -> NaN vector when nothing is within epsilon (accepted).
   - Overflow fallback: stackless equivalent (skip subtrees whose box
     distance_sq >= 1e-6f).
4. `signed_distance_watertight(point, nodes, triangles, max_distance_sq
   = MAX_DIST_SQ) -> std::pair<int, float>`:
   closest triangle (2), its `closest_point(point)` (kept member), the
   average normal (3) AT that closest point, and the result is the
   closest distance with the sign of `avg_normal . (point -
   closest_point)` (copysignf; sign of a zero/NaN dot follows copysignf
   semantics naturally).
5. `signed_distance_raystab(point, nodes, triangles, max_distance_sq =
   MAX_DIST_SQ, pcg32 rng = {}) -> std::pair<int, float>`
   (Nooruddin-Turk stabbing with a randomized Fibonacci direction set):
   - Closest triangle first (for the magnitude and face index).
   - `Vector2f offset = {rng.next_float(), rng.next_float()};` -- pcg32
     from `<gpu/pcg32.h>` (kept third-party header), exactly two draws,
     x then y.
   - 32 directions `fibonacci_dir<32>(i, offset)`, i = 0..31. For each
     direction d: if EITHER the ray from `point` along `-d` OR along
     `+d` hits nothing (ray_intersect index < 0), the point sees the
     outside -> return the POSITIVE distance immediately.
   - If every direction is blocked both ways, return the NEGATIVE
     distance (inside).

### Kernels and wrappers

Four `__global__` kernels (thread i = element i, guard `i >= n`). You may
unify their shape (e.g. a templated kernel over a query functor) or write
four plain kernels -- your choice. Behavior:

- unsigned distance: `closest_triangle` with the default bound
  (a `use_existing_distances_as_upper_bounds` bool parameter selects
  `distances[i]` as the bound instead; the wrappers pass false -- keep
  the parameter).
  Writes `distances[i]` (unsquared), `face_id[i] = triangles[winner].id`
  (the ORIGINAL face index survives the build's reordering via the id
  field), and, when the uvw pointer is non-null,
  `uvw[i] = winner.barycentric(winner.closest_point(point))` (kept
  members).
- signed distance watertight / raystab: same outputs, distance carries
  the sign. Raystab kernel: `pcg32 rng; rng.advance(i * 2);` before the
  query -- the per-element stream offset is pinned (2 draws per element).
- ray trace: `ray_intersect`; writes `depth[i] = t` (MAX_DIST on miss),
  `positions[i] = ro + t * rd` (so misses land at distance MAX_DIST
  along the ray), `face_id[i] = triangles[winner].id` on a hit, else -1.

Wrapper virtuals (`*_gpu`): cast the raw float pointers to
`Eigen::Vector3f*` / const variants, lazily upload the node array to the
GPU on first use (`if (m_nodes_gpu.data() == nullptr)
m_nodes_gpu.resize_and_copy_from_host(m_nodes);` -- the lazy behavior is
deliberate and must stay), then launch the kernel with the 128-thread
helper on the caller's stream. `mode` 0 = watertight, else raystab.

`TriangleBvh::make()` returns the concrete instance.

## What "independent" means here

Same behavior, different text. Use your own decomposition, naming, and
control flow; where this spec pins an exact formula (sentinels, epsilons,
the Fibonacci mapping, pcg32 draws) reproduce the VALUES, not any
particular source text. If you find only one natural spelling for a
3-line function, prefer a genuinely different decomposition (helper
functions, different loop shape) and note it in your report. Do not
consult the previous implementations or instant-ngp; everything you need
is in this spec, the kept snippets, and the published references.

## Acceptance

- `PYTORCH_ROCM_ARCH=gfx90a pip install -e . --no-build-isolation` (from
  the repo root, ROCm torch) compiles clean.
- `python projects/cubvh/harness/golden.py check --ref <goldens>` and
  `... crossload --ref <goldens>` both PASS (the crossload check loads
  serialized node arrays from the OLD build -- this is the
  compatibility contract for the node encoding).
- `python test/signed_distance.py`, `test/unsigned_distance.py`,
  `test/state_dict.py` pass as before.
